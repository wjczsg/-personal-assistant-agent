# 导入类型提示工具。
import logging
from typing import Any
from uuid import uuid4
# 导入配置、记忆和工具函数。
from ..config import settings
from ..llm.client import llm_client
from .agent_state import AgentRunState
from .clarification_service import (
    clarification_question,
    finish_pending_clarification,
    remember_clarification,
    resolve_pending_clarification,
)
from .context_budget import ContextBudgetExceeded
from .context_service import build_model_context, prepare_agent_context
from .conversation_service import finish_conversation_turn, start_conversation_turn
from .routing_service import decide_tool_route
from .tool_executor import (
    MODEL_TOOL_NAMES,
    ToolExecutionResult,
    execute_model_tool as run_model_tool,
    execute_tool_call,
)
# 待办确认/取消的识别和数据库操作已经独立到服务模块；Agent 只负责调用。
from .todo_proposal_service import format_proposal_reply, handle_todo_reply

logger = logging.getLogger("uvicorn.error")
# 这个类就是项目里的“个人助理 Agent”。
class PersonalAssistantAgent:
    # 初始化 Agent，并保存本次对话所属的用户。
    def __init__(self, user_id: str):
        """创建一个绑定当前用户的 Agent。

        Agent 对象只长期保存 user_id；工具调用、提案、历史和摘要等数据都放在
        每次 run() 新建的 AgentRunState 中，避免不同请求共享临时状态。
        """

        # 保存用户编号，后面查询记忆和待办时要用它。
        self.user_id = user_id

    # 统一记录工具调用，便于返回给前端学习观察。
    def _record_tool(self, state: AgentRunState, name: str, result: str) -> str:
        """把一次真实工具执行记录到本轮状态，并原样返回工具结果。

        state 是当前请求独有的状态；name 和 result 会返回给前端，用于显示
        “本轮使用了哪些工具”。返回 result 可以让调用方继续组织模型上下文。
        """

        # 把工具名称和工具结果保存起来。
        state.tool_calls.append({"name": name, "result": result})
        # 把结果继续交给后续回答流程。
        return result

    # 根据用户文字做几个适合初学者的工具调用判断。
    async def _try_tools(self, message: str, state: AgentRunState) -> str | None:
        """保留旧的关键词工具调用入口，供学习新旧方案时对照。

        当前所有业务工具都由模型通过 Function Calling 选择，因此这个方法
        正常返回 None。state 参数为将来重新启用本地工具规则时预留。
        """

        # 下面保留旧的正则计算代码，注释掉是为了方便你对比新旧两种方案。
        # 旧方案需要 Python 自己从用户原话里匹配“12 * 8”。
        # match = re.search(r"(?:算|计算)\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)", message)
        # if match:
        #     a = float(match.group(1))
        #     operator = match.group(2)
        #     b = float(match.group(3))
        #     return self._record_tool(state, "calculator", calculator(a, b, operator))
        # 保留旧的天气关键词和正则提取逻辑供学习对照；现在由模型提取 city 参数。
        # if any(keyword in message for keyword in ["天气", "气温", "下雨", "降温", "温度"]):
        #     city_match = re.search(
        #         r"(?:帮我)?(?:查询|查一下|查|看看|告诉我|问一下)?\s*"
        #         r"([\u4e00-\u9fa5A-Za-z·]{2,20}?)\s*"
        #         r"(?:今天|明天|现在|当前)?(?:的)?(?:天气|气温|温度)",
        #         message,
        #     )
        #     if not city_match:
        #         return self._record_tool(state, "get_weather", "请告诉我想查询的城市，例如：查询北京天气。")
        #     city = city_match.group(1).strip()
        #     return self._record_tool(state, "get_weather", await execute_tool("get_weather", {"city": city}, self.user_id))
        # 保留旧的时间关键词判断供学习对照；现在由模型决定是否调用时间工具。
        # if "几点" in message or "时间" in message or "日期" in message:
        #     return self._record_tool(state, "get_current_time", await execute_tool("get_current_time", {}, self.user_id))
        # 保留旧的待办关键词判断供学习对照；现在由模型决定是否调用 get_todos。
        # if "我的待办" in message or "有哪些待办" in message or "待办事项" in message:
        #     return self._record_tool(state, "get_todos", await execute_tool("get_todos", {}, self.user_id))
        # 保留旧的“记住/添加待办”关键词判断供学习对照；现在由模型提取 content 参数。
        # if "记住" in message or "添加待办" in message:
        #     content = re.sub(r"^(帮我)?(记住|添加待办)[:：\s]*", "", message).strip()
        #     if not content:
        #         return self._record_tool("remember_todo", "请告诉我具体要记住什么。")
        #     return self._record_tool(state, "remember_todo", await execute_tool("remember_todo", {"content": content}, self.user_id))
        # 没有需要调用其他工具时返回 None。
        return None

    def _apply_tool_execution(
        self,
        state: AgentRunState,
        execution: ToolExecutionResult,
    ) -> str:
        """把无状态执行器返回的结果同步到当前 Agent 请求状态。"""

        # propose_todo 成功时保留真实提案，最终回答必须提示用户后续确认。
        if execution.proposal is not None:
            state.proposed_todos.append(execution.proposal)
        # 只有真实工具函数已经运行时才展示在前端“已使用工具”列表中。
        if execution.executed:
            return self._record_tool(state, execution.name, execution.content)
        return execution.content

    # 保留这个薄包装，兼容项目现有调用；校验与执行逻辑已经移到 tool_executor.py。
    async def _execute_model_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        allowed_tool_names: tuple[str, ...] | None = None,
        state: AgentRunState | None = None,
    ) -> str:
        """执行一个模型指定的工具，并把结果同步到请求状态。

        这是兼容现有测试和旧调用方式的薄包装。正式的多工具循环使用
        execute_tool_call()；直接调用该方法但没有传 state 时，会创建隔离状态。
        """

        # 兼容直接调用这个私有方法的测试；正式 run() 会显式传入本轮状态。
        if state is None:
            state = AgentRunState()
        execution = await run_model_tool(
            name=name,
            arguments=arguments,
            user_id=self.user_id,
            allowed_tool_names=allowed_tool_names,
        )
        return self._apply_tool_execution(state, execution)

    # 调用大模型，让它理解问题，并在需要时选择已注册的工具。
    async def _ask_llm(
        self,
        message: str,
        memory: list[dict[str, Any]],
        state: AgentRunState | None = None,
    ) -> str:
        """执行一次可能包含多轮 Function Calling 的模型对话。

        先完成工具路由和上下文准备，再循环执行“请求模型 → 执行工具 →
        带工具结果重新请求模型”，直到得到最终文字或达到最大工具轮数。
        memory 是本轮开始前的普通聊天历史，state 保存本轮产生的临时数据。
        """

        # 正式流程由 run() 传入；直接测试该私有方法时创建一份独立临时状态。
        if state is None:
            state = AgentRunState()
        # 没有配置 API 密钥时使用演示模式，避免初学者一开始就被配置卡住。
        if not settings.llm_api_key:
            # 把历史记忆压缩成一段文字。
            memory_text = "；".join(item["content"] for item in memory[-4:])
            # 返回一个简单但能体现记忆的演示回答。
            return f"演示模式回答：我收到了“{message}”。最近记忆：{memory_text or '暂时没有历史记录'}"
        # 本地规则、轻量模型兜底和只读工具限制都由 routing_service 统一处理。
        route_decision = await decide_tool_route(message)
        # 上下文服务统一准备提示词、长期记忆、工具定义、历史和摘要状态。
        context_state = await prepare_agent_context(
            user_id=self.user_id,
            message=message,
            route_decision=route_decision,
            history=memory,
            history_records=state.history_records,
            history_upper_id=state.history_upper_id,
            summary_state=state.summary_state,
        )
        selected_tool_names = context_state.allowed_tool_names
        tool_messages: list[dict[str, Any]] = []
        # 一轮聊天中的多次模型请求使用同一编号，便于从终端还原工具调用顺序。
        run_id = uuid4().hex[:8]
        # 限制工具调用轮数，既支持多工具连续调用，也防止模型异常时无限循环。
        max_tool_rounds = 3
        # 按轮次请求模型，直到模型返回最终文字。
        for round_no in range(1, max_tool_rounds + 1):
            # 工具结果会逐轮增加，因此每次请求模型前都让上下文服务重新算预算。
            try:
                budget_result = await build_model_context(
                    state=context_state,
                    tool_messages=tool_messages,
                )
            except ContextBudgetExceeded:
                results = [item for item in tool_messages if item.get("role") == "tool"]
                logger.warning(
                    "Agent 上下文超预算：run=%s round=%d budget=%d tool_results=%d result_chars=%d",
                    run_id, round_no, context_state.input_limit, len(results),
                    sum(len(item.get("content") or "") for item in results),
                )
                raise
            # Agent 只描述本轮需要什么；URL、鉴权和 HTTP 请求统一交给 llm_client。
            data = await llm_client.complete(
                messages=budget_result.messages,
                temperature=0.7,
                tools=context_state.tool_definitions or None,
                tool_choice="auto" if context_state.tool_definitions else None,
                timeout=60,
            )
            # 取出本轮模型生成的 assistant 消息。
            assistant_message = data["choices"][0]["message"]
            # 读取模型是否提出了工具调用请求。
            tool_calls = assistant_message.get("tool_calls") or []
            if tool_calls:
                tool_names = []
                for item in tool_calls:
                    function = item.get("function")
                    tool_names.append(
                        function.get("name", "<unknown>")
                        if isinstance(function, dict) else "<unknown>"
                    )
                logger.info(
                    "Agent 工具请求：run=%s round=%d tools=%s",
                    run_id, round_no, tool_names,
                )
            # 如果没有工具调用，说明模型已经生成最终答案。
            if not tool_calls:
                # 返回模型生成的自然语言；content 为空时给出兜底文字。
                answer = assistant_message.get("content") or "我暂时没有生成有效回答。"
                question = clarification_question(answer)
                if question is not None:
                    # 有待补充的业务任务才跨轮保存；普通闲聊的追问仍走对话历史。
                    if (route_decision.tool_names or route_decision.routes != ("chat",)) and not state.proposed_todos:
                        await remember_clarification(self.user_id, message, question)
                        state.clarification_saved = True
                    return question
                return answer
            # 把模型的工具调用消息放回上下文，下一轮模型才能知道自己刚才请求了什么。
            tool_messages.append(assistant_message)
            # 逐个处理模型提出的工具调用。
            for tool_call in tool_calls:
                # 原始 JSON 参数解析、权限校验和注册表调用统一交给工具执行器。
                execution = await execute_tool_call(
                    tool_call=tool_call,
                    user_id=self.user_id,
                    allowed_tool_names=selected_tool_names,
                )
                # Agent 只接收结构化结果并维护本轮状态，不再了解工具执行细节。
                self._apply_tool_execution(state, execution)
                tool_messages.append(execution.to_tool_message())
                logger.info(
                    "Agent 工具结果：run=%s round=%d tool=%s executed=%s result_chars=%d",
                    run_id, round_no, execution.name, execution.executed, len(execution.content),
                )
        # 达到最大调用轮数仍未得到最终答案时，返回明确提示。
        return "工具调用次数超过限制，请稍后再试。"

    # Agent 对外提供的主入口。
    async def run(self, message: str) -> tuple[str, list[dict[str, str]], int]:
        """处理一条用户消息并返回回答、工具记录和读取到的历史数量。

        这个方法负责整轮编排：建立旧历史快照、创建请求状态、处理待办确认、
        调用模型工具循环、处理异常，最后把助手回答保存到 SQLite。
        """

        # 对话服务先建立旧历史快照，再保存当前问题，避免未回答问题混入旧历史摘要。
        turn = await start_conversation_turn(
            user_id=self.user_id,
            message=message,
        )
        memory = turn.history
        # 每条消息都创建独立状态，并装入本轮开始时读取的历史与摘要快照。
        # 这些数据只服务当前请求，不再写进可被后续请求复用的 Agent 对象。
        state = AgentRunState(
            history_records=list(turn.records),
            history_upper_id=turn.history_upper_id,
            summary_state=dict(turn.summary_state),
        )
        # 在普通路由之前恢复上轮等待补充的任务；新任务或取消会清除旧状态。
        pending_resolution = await resolve_pending_clarification(self.user_id, message)
        if pending_resolution.direct_answer is not None:
            await finish_conversation_turn(user_id=self.user_id, answer=pending_resolution.direct_answer)
            return pending_resolution.direct_answer, state.tool_calls, turn.history_count
        effective_message = pending_resolution.message
        # 确认/取消是新一条用户消息触发的后端操作，不让模型在提案当轮替用户批准。
        decision = await handle_todo_reply(self.user_id, message)
        # 返回 None 表示这不是确认/取消消息，后面继续正常的模型对话流程。
        if decision is not None:
            # 只有真正确认或取消成功时才记录工具；选择提示不伪装成数据库操作。
            if decision.tool_name is not None:
                self._record_tool(state, decision.tool_name, decision.answer)
            # 服务已经生成最终回答，这条分支不需要再调用大模型。
            await finish_conversation_turn(user_id=self.user_id, answer=decision.answer)
            return decision.answer, state.tool_calls, turn.history_count
        # 目前所有工具都交给模型通过 Function Calling 选择。
        tool_result = await self._try_tools(effective_message, state)
        # 如果调用了工具，就直接把工具结果组织成回答。
        if tool_result is not None:
            # 保存 Agent 的回答，下一次对话就能记住它。
            answer = f"工具已经帮你处理好了：\n{tool_result}"
            # 由对话服务统一保存回答。
            await finish_conversation_turn(user_id=self.user_id, answer=answer)
            # 返回回答、工具调用记录和记忆数量。
            return answer, state.tool_calls, turn.history_count
        # 普通问题交给大模型处理。
        try:
            answer = await self._ask_llm(effective_message, memory, state)
        except ContextBudgetExceeded as exc:
            # 必需内容本身超限时给出可操作提示，不继续发送注定超限的请求。
            if state.proposed_todos:
                answer = format_proposal_reply(state.proposed_todos)
            elif state.tool_calls:
                # 工具可能已经修改数据，不能让用户误以为操作没有发生并再次提交。
                tool_names = "、".join(item["name"] for item in state.tool_calls)
                answer = (
                    f"本轮工具已执行（{tool_names}），但后续模型请求超出上下文预算。"
                    "请先查看下方工具结果；不要直接重复执行可能修改数据的操作。"
                )
            else:
                answer = str(exc)
        except Exception:
            # 提案已保存时，即使模型后续生成回答失败，也要把提案编号明确交给用户。
            if not state.proposed_todos:
                raise
            answer = format_proposal_reply(state.proposed_todos)
        else:
            if state.proposed_todos:
                # 使用数据库工具返回的真实提案覆盖模型措辞，防止误报“已添加”。
                answer = format_proposal_reply(state.proposed_todos)
        if pending_resolution.resumed and not state.clarification_saved:
            # 原任务已完成；如果模型再次追问，新状态已经由 _ask_llm 更新，不清除。
            await finish_pending_clarification(self.user_id)
        # 对话服务统一完成这一轮，把最终回答保存为 assistant 消息。
        await finish_conversation_turn(user_id=self.user_id, answer=answer)
        # 返回回答、工具调用记录和记忆数量。
        return answer, state.tool_calls, turn.history_count
