# 导入 asyncio，用线程执行同步的 Chroma 检索，避免阻塞异步请求。
import asyncio
import logging
# 导入 json，用来读取大模型返回的 JSON 内容。
import json
# 导入类型提示工具。
from typing import Any
# 导入配置、记忆和工具函数。
from ..config import settings
from ..llm.client import llm_client
from .context_budget import ContextBudgetExceeded, build_context
from ..storage.database import (
    get_conversation_summary,
    get_recent_message_records,
    save_message,
)
from ..tooling.tool_registry import get_tool_definitions
from .tool_router import (
    FALLBACK_READ_ONLY_TOOLS,
    ROUTE_ORDER,
    ToolRouteDecision,
    select_tool_route,
    should_use_fallback_classifier,
)
from .prompts import build_system_prompt
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
from ..storage.vector_store import search_memories

# 使用 Uvicorn 的应用日志通道，确保通过 PyCharm 启动服务时，
# 轻量路由结果会和访问日志一起显示在当前终端中。
logger = logging.getLogger("uvicorn.error")

# 只有问题可能涉及用户长期背景时，才尝试查询长期记忆。
# 这不是最终的语义判断，而是一个低成本的“是否值得检索”门控，
# 用来避免“你好”、天气和待办等问题每次都访问 Chroma。
MEMORY_QUERY_HINTS = (
    "长期记忆",
    "记得我",
    "之前说过",
    "上次说过",
    "我的目标",
    "职业目标",
    "求职方向",
    "工作方向",
    "发展方向",
    "想从事",
    "毕业后",
    "我的计划",
    "我的背景",
    "我的经历",
    "我的偏好",
    "我喜欢",
    "我习惯",
    "按照我的",
    "根据我的目标",
)


def should_search_memory(message: str) -> bool:
    """判断当前问题是否值得查询长期记忆。

    这里使用轻量关键词门控，而不是额外请求一次大模型：额外调用模型会增加成本和延迟，
    也不适合作为上下文优化的第一步。后续如果规则覆盖不足，可以替换成独立分类器，
    但不需要改变 _get_relevant_memories 的调用接口。
    """
    normalized_message = message.strip()
    if not normalized_message:
        return False
    return any(hint in normalized_message for hint in MEMORY_QUERY_HINTS)

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

    # 从 Chroma 检索与当前问题相关的长期记忆。
    async def _get_relevant_memories(
        self,
        message: str,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        # 普通问候、天气、计算和待办问题通常不依赖长期个人背景，直接跳过向量检索。
        # force=True 只由轻量路由判断为 memory 时使用，避免路由结果被门控规则再次挡住。
        if not force and not should_search_memory(message):
            return []
        # Chroma 当前使用同步 API；放到线程中执行，避免阻塞 FastAPI 的异步事件循环。
        try:
            return await asyncio.to_thread(
                search_memories,
                self.user_id,
                message,
                3,
                settings.memory_distance_threshold,
            )
        # 向量数据库暂时不可用时，继续进行普通对话，不让记忆功能拖垮 Agent。
        except Exception:
            return []

    async def _classify_fallback_route(self, message: str) -> str | None:
        """对规则未识别出的具体问题做一次低成本、只输出类别的判断。"""

        if not settings.llm_api_key or not should_use_fallback_classifier(message):
            return None

        classifier_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是工具路由分类器。只根据用户当前问题选择一个类别，"
                    "不要回答问题，不要调用工具。只能输出 JSON："
                    '{"route":"weather|todo|document|memory|chat"}。'
                    "weather 表示天气或天气影响，todo 表示查询待办，"
                    "document 表示询问已上传资料，memory 表示询问用户个人背景，"
                    "chat 表示其他问题。无法确定时选择 chat。"
                ),
            },
            {"role": "user", "content": message},
        ]
        # DeepSeek 的思考内容对这个分类没有价值，关闭后能减少延迟和消耗。
        thinking = {"type": "disabled"} if settings.llm_model.startswith("deepseek-") else None

        try:
            data = await llm_client.complete(
                messages=classifier_messages,
                temperature=0,
                # 分类结果只有一个枚举值，32 个输出 token 已经足够。
                max_tokens=32,
                response_format={"type": "json_object"},
                thinking=thinking,
                timeout=20,
            )
            content = data["choices"][0]["message"].get("content")
            parsed = json.loads(content) if isinstance(content, str) else {}
            route = parsed.get("route") if isinstance(parsed, dict) else None
            if route not in ROUTE_ORDER:
                return None
            logger.info("轻量工具路由结果：route=%s", route)
            return route
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("轻量工具路由响应格式无效：%s", exc)
        except Exception as exc:
            # 分类失败不能阻断正常聊天；后续仍按 chat 路由请求主模型。
            logger.warning("轻量工具路由调用失败：%s", exc)
        return None

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
        # 先选择本轮业务路由；提示词模块需要根据结果写明路由和工具白名单。
        route_decision = select_tool_route(
            message,
            needs_memory=should_search_memory(message),
        )

        # 只有规则没有选出任何工具时，才额外请求一次轻量分类器。
        # 直接命中天气、待办或文档规则的问题不会增加第二次模型请求。
        if not route_decision.tool_names and route_decision.routes == ("chat",):
            fallback_route = await self._classify_fallback_route(message)
            if fallback_route is not None:
                route_decision = ToolRouteDecision(
                    routes=(fallback_route,),
                    tool_names=FALLBACK_READ_ONLY_TOOLS[fallback_route],
                )
        selected_tool_names = route_decision.tool_names
        # Agent 主流程不再维护大段提示词；具体文字规则统一由 prompts.py 组装。
        system_message = {
            "role": "system",
            "content": build_system_prompt(
                routes=route_decision.routes,
                tool_names=selected_tool_names,
            ),
        }
        # 从 Chroma 查询与当前问题语义相关的长期记忆。
        relevant_memories = await self._get_relevant_memories(
            message,
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
        # 查询这个用户最近的消息，作为 Agent 的记忆。
        # 多取一些历史作为候选，再由输入预算决定本轮实际发送多少。
        # 【摘要流程 1/6】先读旧聊天，再保存本次问题；本次未回答的问题不能当作旧问答总结。
        records = get_recent_message_records(self.user_id, limit=100)
        memory = [{"role": row["role"], "content": row["content"]} for row in records]
        # 先保存用户本次输入。
        save_message(self.user_id, "user", message)
        # 确认/取消是新一条用户消息触发的后端操作，不让模型在提案当轮替用户批准。
        decision = await handle_todo_reply(self.user_id, message)
        # 返回 None 表示这不是确认/取消消息，后面继续正常的模型对话流程。
        if decision is not None:
            # 只有真正确认或取消成功时才记录工具；选择提示不伪装成数据库操作。
            if decision.tool_name is not None:
                self._record_tool(decision.tool_name, decision.answer)
            # 服务已经生成最终回答，这条分支不需要再调用大模型。
            save_message(self.user_id, "assistant", decision.answer)
            return decision.answer, self.tool_calls, len(memory)
        # 目前所有工具都交给模型通过 Function Calling 选择。
        tool_result = await self._try_tools(message)
        # 如果调用了工具，就直接把工具结果组织成回答。
        if tool_result is not None:
            # 保存 Agent 的回答，下一次对话就能记住它。
            answer = f"工具已经帮你处理好了：\n{tool_result}"
            # 保存回答。
            save_message(self.user_id, "assistant", answer)
            # 返回回答、工具调用记录和记忆数量。
            return answer, self.tool_calls, len(memory)
        # 普通问题交给大模型处理。
        # 【摘要流程 1/6】记住旧聊天的末尾 ID，并读取上次保存的摘要及处理进度。
        self._history_records = records
        self._history_upper_id = int(records[-1]["id"]) if records else 0
        self._summary_state = get_conversation_summary(self.user_id)
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
        # 保存模型回答。
        save_message(self.user_id, "assistant", answer)
        # 返回回答、工具调用记录和记忆数量。
        return answer, self.tool_calls, len(memory)
