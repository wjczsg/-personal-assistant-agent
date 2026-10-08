# 导入 datetime，用来返回当前时间。
from datetime import datetime
# 导入 database 文件中的待办操作。
from .database import add_todo, list_todos

# 计算器工具：把两个数字相加。
def calculator(a: float, b: float, operator: str) -> str:
    # 根据 operator 的值决定使用哪一种计算方法。
    if operator == "+":
        # 返回加法结果。
        return str(a + b)
    # 如果用户要求减法，就返回减法结果。
    if operator == "-":
        # 返回减法结果。
        return str(a - b)
    # 如果用户要求乘法，就返回乘法结果。
    if operator == "*":
        # 返回乘法结果。
        return str(a * b)
    # 如果用户要求除法，就先检查除数是不是零。
    if operator == "/":
        # 除数为零时直接返回易懂的错误。
        if b == 0:
            return "除数不能为 0"
        # 返回除法结果。
        return str(a / b)
    # 没有匹配的运算符时返回提示。
    return "暂时只支持 +、-、*、/"

# 时间工具：返回服务器当前时间。
def get_current_time() -> str:
    # 把当前时间格式化成人能看懂的文字。
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 记事工具：把用户想记住的内容保存到数据库。
def remember_todo(user_id: str, content: str) -> str:
    # 调用数据库函数新增待办。
    todo = add_todo(user_id, content)
    # 返回一条给 Agent 看的结果。
    return f"已记录待办 #{todo['id']}：{todo['content']}"

# 待办查询工具：读取用户还没有完成的事情。
def get_todos(user_id: str) -> str:
    # 从数据库中读取待办列表。
    todos = list_todos(user_id)
    # 没有待办时返回明确提示。
    if not todos:
        return "目前没有未完成的待办事项。"
    # 把每条待办整理成一行。
    lines = [f"#{todo['id']}：{todo['content']}" for todo in todos]
    # 把多行内容合并成一段文字。
    return "\n".join(lines)
