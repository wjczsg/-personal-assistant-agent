# 从 pathlib 导入 Path，方便我们用跨平台的方式处理文件路径。
from pathlib import Path
# 从 pydantic_settings 导入 BaseSettings，让程序可以读取 .env 配置文件。
from pydantic_settings import BaseSettings, SettingsConfigDict

# 定义整个项目的配置对象。
class Settings(BaseSettings):
    # 大模型服务的基础地址。
    llm_base_url: str = "https://api.openai.com/v1"
    # 大模型 API 密钥；空字符串代表使用演示模式。
    llm_api_key: str = ""
    # 要调用的模型名称。
    llm_model: str = "gpt-4o-mini"
    # SQLite 数据库文件名。
    database_file: str = "assistant.db"

    # 告诉 Pydantic 从项目根目录的 .env 文件读取配置。
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# 创建一个全局配置对象，其他文件可以直接导入它。
settings = Settings()
# 把数据库文件放在项目根目录，而不是放在 app 文件夹里面。
DATABASE_PATH = Path(__file__).resolve().parent.parent / settings.database_file
