# 导入类型提示工具。
from typing import Any
# 导入配置、记忆和工具函数。
from ..config import settings
from ..llm.client import llm_client
from .context_budget import ContextBudgetExceeded, build_context
from .conversation_service import finish_conversation_turn, start_conversation_turn
from ..tooling.tool_registry import get_tool_definitions
from .prompts import build_system_prompt
from .memory_service import search_relevant_memories
from .routing_service import decide_tool_route
from .tool_executor import (
    MODEL_TOOL_NAMES,
    ToolExecutionResult,
    execute_model_tool as run_model_tool,
    execute_tool_call,
)
# 待办确认/取消的识别和数据库操作已经独立到服务模块；Agent 只负责调用。
from .todo_proposal_service import format_proposal_reply, handle_todo_reply
from .history_summary import (
    SUMMARY_BATCHES_PER_REQUEST,
    refresh_summary_if_needed,
    summary_message,
)
# 这个类就是项目里的“个人助理 Agent”。
class PersonalAssistantAgent:
    # 初始化 Agent，并保存本次对话所属的用户。
    def __init__(self, user_id: str):
        # 保存用户编号，后面查询记忆和待办时要用它。
        self.user_id = user_id
        # 用列表记录这一次请求实际调用了哪些工具。
        self.tool_calls: list[dict[str, str]] = []
        # 只记录本轮由模型新建的提案；稍后用数据库里的真实内容生成确认提示。
        self.proposed_todos: list[dict[str, Any]] = []
        # run() 中获取的带编号历史只供摘要定位；编号绝不作为聊天消息发给模型。
        self._history_records: list[dict[str, Any]] = []
        self._history_upper_id = 0
        self._summary_state: dict[str, Any] | None = None

    # 统一记录工具调用，便于返回给前端学习观察。
    def _record_tool(self, name: str, result: str) -> str:
        # 把工具名称和工具结果保存起来。
        self.tool_calls.append({"name": name, "result": result})
        # 把结果继续交给后续回答流程。
        return result

    # 根据用户文字做几个适合初学者的工具调用判断。
    async def _try_tools(self, message: str) -> str | None:
        # 下面保留旧的正则计算代码，注释掉是为了方便你对比新旧两种方案。
        # 旧方案需要 Python 自己从用户原话里匹配“12 * 8”。
        # match = re.search(r"(?:算|计算)\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)", message)
        # if match:
        #     a = float(match.group(1))
        #     operator = match.group(2)
        #     b = float(match.group(3))
        #     return self._record_tool("calculator", calculator(a, b, operator))
        # 保留旧的天气关键词和正则提取逻辑供学习对照；现在由模型提取 city 参数。
        # if any(keyword in message for keyword in ["天气", "气温", "下雨", "降温", "温度"]):
        #     city_match = re.search(
        #         r"(?:帮我)?(?:查询|查一下|查|看看|告诉我|问一下)?\s*"
        #         r"([\u4e00-\u9fa5A-Za-z·]{2,20}?)\s*"
        #         r"(?:今天|明天|现在|当前)?(?:的)?(?:天气|气温|温度)",
        #         message,
        #     )
        #     if not city_match:
        #         return self._record_tool("get_weather", "请告诉我想查询的城市，例如：查询北京天气。")
        #     city = city_match.group(1).strip()
        #     return self._record_tool("get_weather", await execute_tool("get_weather", {"city": city}, self.user_id))
        # 保留旧的时间关键词判断供学习对照；现在由模型决定是否调用时间工具。
        # if "几点" in message or "时间" in message or "日期" in message:
        #     return self._record_tool("get_current_time", await execute_tool("get_current_time", {}, self.user_id))
        # 保留旧的待办关键词判断供学习对照；现在由模型决定是否调用 get_todos。
        # if "我的待办" in message or "有哪些待办" in message or "待办事项" in message:
        #     return self._record_tool("get_todos", await execute_tool("get_todos", {}, self.user_id))
        # 保留旧的“记住/添加待办”关键词判断供学习对照；现在由模型提取 content 参数。
        # if "记住" in message or "添加待办" in message:
        #     content = re.sub(r"^(帮我)?(记住|添加待办)[:：\s]*", "", message).strip()
        #     if not content:
        #         return self._record_tool("remember_todo", "请告诉我具体要记住什么。")
        #     return self._record_tool("remember_todo", await execute_tool("remember_todo", {"content": content}, self.user_id))
        # 没有需要调用其他工具时返回 None。
        return None

    def _apply_tool_execution(self, execution: ToolExecutionResult) -> str:
        """把无状态执行器返回的结果同步到当前 Agent 请求状态。"""

        # propose_todo 成功时保留真实提案，最终回答必须提示用户后续确认。
        if execution.proposal is not None:
            self.proposed_todos.append(execution.proposal)
        # 只有真实工具函数已经运行时才展示在前端“已使用工具”列表中。
        if execution.executed:
            return self._record_tool(execution.name, execution.content)
        return execution.content

    # 保留这个薄包装，兼容项目现有调用；校验与执行逻辑已经移到 tool_executor.py。
    async def _execute_model_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        allowed_tool_names: tuple[str, ...] | None = None,
    ) -> str:
        execution = await run_model_tool(
            name=name,
            arguments=arguments,
            user_id=self.user_id,
            allowed_tool_names=allowed_tool_names,
        )
        return self._apply_tool_execution(execution)

    # 调用大模型，让它理解问题，并在需要时选择已注册的工具。
    async def _ask_llm(self, message: str, memory: list[dict[str, Any]]) -> str:
        # 没有配置 API 密钥时使用演示模式，避免初学者一开始就被配置卡住。
        if not settings.llm_api_key:
            # 把历史记忆压缩成一段文字。
            memory_text = "；".join(item["content"] for item in memory[-4:])
            # 返回一个简单但能体现记忆的演示回答。
            return f"演示模式回答：我收到了“{message}”。最近记忆：{memory_text or '暂时没有历史记录'}"
        # 本地规则、轻量模型兜底和只读工具限制都由 routing_service 统一处理。
        route_decision = await decide_tool_route(message)
        selected_tool_names = route_decision.tool_names
        # Agent 主流程不再维护大段提示词；具体文字规则统一由 prompts.py 组装。
        system_message = {
            "role": "system",
            "content": build_system_prompt(
                routes=route_decision.routes,
                tool_names=selected_tool_names,
            ),
        }
        # 记忆门控、Chroma 查询和故障降级都由 memory_service 负责。
        relevant_memories = await search_relevant_memories(
            user_id=self.user_id,
            message=message,
            force="memory" in route_decision.routes,
        )
        memory_messages: list[dict[str, Any]] = []
        # 只有检索到长期记忆时，才额外加入长期记忆上下文。
        if relevant_memories:
            # 把每条记忆整理成模型容易理解的文本。
            memory_lines = [
                f"- ({item.get('memory_type', 'unknown')}) {item['content']}"
                for item in relevant_memories
            ]
            # 把长期记忆作为独立的系统消息加入上下文。
            memory_messages.append({
                "role": "system",
                "content": (
                    "以下是从当前用户的长期记忆库中检索到的相关信息，"
                    "只能作为回答参考；如果与用户当前明确表达的内容冲突，"
                    "以当前问题为准：\n"
                    + "\n".join(memory_lines)
                ),
            })
        # 文档片段不在这里提前检索；模型调用后进入本轮工具消息。
        current_message = {"role": "user", "content": message}
        tool_messages: list[dict[str, Any]] = []
        tool_definitions = get_tool_definitions(list(selected_tool_names))
        summary_calls_used = 0
        # 限制工具调用轮数，既支持多工具连续调用，也防止模型异常时无限循环。
        max_tool_rounds = 3
        # 按轮次请求模型，直到模型返回最终文字。
        for _ in range(max_tool_rounds):
            # 本轮工具定义固定；每次循环重新计算的是会增长的消息内容。
            summary_messages: list[dict[str, Any]] = []
            if self._summary_state is not None:
                # 工具结果会增加后续轮次的输入量；每轮都检查是否还有旧历史需要摘要。
                remaining = SUMMARY_BATCHES_PER_REQUEST - summary_calls_used
                # 【摘要流程 2～5/6】每轮先判断是否需要把放不下的旧问答更新为摘要。
                if self._history_upper_id and remaining > 0:
                    self._summary_state, memory, calls_made = await refresh_summary_if_needed(
                        user_id=self.user_id,
                        state=self._summary_state,
                        recent_records=self._history_records,
                        history_upper_id=self._history_upper_id,
                        system_message=system_message,
                        memory_messages=memory_messages,
                        current_message=current_message,
                        tool_messages=tool_messages,
                        tool_definitions=tool_definitions,
                        input_limit=settings.context_input_budget_tokens,
                        max_batches=remaining,
                    )
                    summary_calls_used += calls_made
                if self._summary_state["summary"]:
                    summary_messages.append(summary_message(self._summary_state["summary"]))
            # 每轮都重新选择历史：新产生的工具结果会占用输入预算。
            # 本轮工具调用和结果必须完整保留，因此先从旧历史中腾出空间。
            # 【摘要流程 6/6】把摘要、最近问答原文和当前问题一起放进回答模型的输入。
            budget_result = build_context(
                system_message=system_message,
                memory_messages=memory_messages,
                history=memory,
                current_message=current_message,
                tool_messages=tool_messages,
                tool_definitions=tool_definitions,
                input_limit=settings.context_input_budget_tokens,
                summary_messages=summary_messages,
            )
            # Agent 只描述本轮需要什么；URL、鉴权和 HTTP 请求统一交给 llm_client。
            data = await llm_client.complete(
                messages=budget_result.messages,
                temperature=0.7,
                tools=tool_definitions or None,
                tool_choice="auto" if tool_definitions else None,
                timeout=60,
            )
            # 取出本轮模型生成的 assistant 消息。
            assistant_message = data["choices"][0]["message"]
            # 读取模型是否提出了工具调用请求。
            tool_calls = assistant_message.get("tool_calls") or []
            # 如果没有工具调用，说明模型已经生成最终答案。
            if not tool_calls:
                # 返回模型生成的自然语言；content 为空时给出兜底文字。
                return assistant_message.get("content") or "我暂时没有生成有效回答。"
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
                self._apply_tool_execution(execution)
                tool_messages.append(execution.to_tool_message())
        # 达到最大调用轮数仍未得到最终答案时，返回明确提示。
        return "工具调用次数超过限制，请稍后再试。"

    # Agent 对外提供的主入口。
    async def run(self, message: str) -> tuple[str, list[dict[str, str]], int]:
        # 对话服务先建立旧历史快照，再保存当前问题，避免未回答问题混入旧历史摘要。
        turn = await start_conversation_turn(
            user_id=self.user_id,
            message=message,
        )
        memory = turn.history
        # 摘要模块需要带 ID 的旧记录、旧历史末尾 ID 和此前摘要状态。
        self._history_records = turn.records
        self._history_upper_id = turn.history_upper_id
        self._summary_state = turn.summary_state
        # 确认/取消是新一条用户消息触发的后端操作，不让模型在提案当轮替用户批准。
        decision = await handle_todo_reply(self.user_id, message)
        # 返回 None 表示这不是确认/取消消息，后面继续正常的模型对话流程。
        if decision is not None:
            # 只有真正确认或取消成功时才记录工具；选择提示不伪装成数据库操作。
            if decision.tool_name is not None:
                self._record_tool(decision.tool_name, decision.answer)
            # 服务已经生成最终回答，这条分支不需要再调用大模型。
            await finish_conversation_turn(user_id=self.user_id, answer=decision.answer)
            return decision.answer, self.tool_calls, turn.history_count
        # 目前所有工具都交给模型通过 Function Calling 选择。
        tool_result = await self._try_tools(message)
        # 如果调用了工具，就直接把工具结果组织成回答。
        if tool_result is not None:
            # 保存 Agent 的回答，下一次对话就能记住它。
            answer = f"工具已经帮你处理好了：\n{tool_result}"
            # 由对话服务统一保存回答。
            await finish_conversation_turn(user_id=self.user_id, answer=answer)
            # 返回回答、工具调用记录和记忆数量。
            return answer, self.tool_calls, turn.history_count
        # 普通问题交给大模型处理。
        try:
            answer = await self._ask_llm(message, memory)
        except ContextBudgetExceeded as exc:
            # 必需内容本身超限时给出可操作提示，不继续发送注定超限的请求。
            if self.proposed_todos:
                answer = format_proposal_reply(self.proposed_todos)
            elif self.tool_calls:
                # 工具可能已经修改数据，不能让用户误以为操作没有发生并再次提交。
                tool_names = "、".join(item["name"] for item in self.tool_calls)
                answer = (
                    f"本轮工具已执行（{tool_names}），但后续模型请求超出上下文预算。"
                    "请先查看下方工具结果；不要直接重复执行可能修改数据的操作。"
                )
            else:
                answer = str(exc)
        except Exception:
            # 提案已保存时，即使模型后续生成回答失败，也要把提案编号明确交给用户。
            if not self.proposed_todos:
                raise
            answer = format_proposal_reply(self.proposed_todos)
        else:
            if self.proposed_todos:
                # 使用数据库工具返回的真实提案覆盖模型措辞，防止误报“已添加”。
                answer = format_proposal_reply(self.proposed_todos)
        # 对话服务统一完成这一轮，把最终回答保存为 assistant 消息。
        await finish_conversation_turn(user_id=self.user_id, answer=answer)
        # 返回回答、工具调用记录和记忆数量。
        return answer, self.tool_calls, turn.history_count
