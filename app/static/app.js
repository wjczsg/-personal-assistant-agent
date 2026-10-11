const messageList = document.querySelector("#messageList");
const chatForm = document.querySelector("#chatForm");
const messageInput = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const clearButton = document.querySelector("#clearButton");
const uploadForm = document.querySelector("#uploadForm");
const documentFile = document.querySelector("#documentFile");
const uploadButton = document.querySelector("#uploadButton");
const uploadStatus = document.querySelector("#uploadStatus");
const userId = localStorage.getItem("assistant-user-id") || "web-user";

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = documentFile.files[0];
  if (!file) {
    uploadStatus.textContent = "请先选择 .txt 或 .md 文件。";
    return;
  }
  // 浏览器先给即时反馈；后端会再做一次校验，不能只信浏览器。
  if (!/\.(txt|md)$/i.test(file.name) || file.size > 1024 * 1024) {
    uploadStatus.textContent = "请选择不超过 1 MB 的 .txt 或 .md 文件。";
    return;
  }
  const formData = new FormData();
  formData.append("user_id", userId);
  formData.append("file", file);
  uploadButton.disabled = true;
  uploadStatus.textContent = "正在保存文档……";
  try {
    const response = await fetch("/api/documents/upload", { method: "POST", body: formData });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "上传失败");
    // 显示文档编号，方便下一步在 /docs 中对这篇文档手动建立索引。
    uploadStatus.textContent = `已保存「${result.filename}」（${result.char_count} 字，编号 #${result.document_id}）；文档索引需在 /docs 中建立。`;
    documentFile.value = "";
  } catch (error) {
    uploadStatus.textContent = `保存失败：${error.message}`;
  } finally {
    uploadButton.disabled = false;
  }
});

function addMessage(role, text, toolCalls = [], memoryCount = null) {
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "我" : "ʕ•ᴥ•ʔ";
  const content = document.createElement("div");
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  content.appendChild(bubble);

  if (role === "assistant" && toolCalls.length) {
    const toolList = document.createElement("div");
    toolList.className = "tool-list";
    toolCalls.forEach((tool) => {
      const tag = document.createElement("span");
      tag.className = "tool-tag";
      tag.textContent = `已使用：${tool.name}`;
      toolList.appendChild(tag);
    });
    content.appendChild(toolList);
  }

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = role === "assistant" && memoryCount !== null
    ? `记忆中有 ${memoryCount} 条最近消息`
    : role === "user" ? "刚刚" : "小熊助手";
  content.appendChild(meta);
  wrapper.append(avatar, content);
  messageList.appendChild(wrapper);
  messageList.scrollTop = messageList.scrollHeight;
}

function addTyping() {
  const node = document.createElement("article");
  node.className = "message assistant typing";
  node.innerHTML = '<div class="avatar">ʕ•ᴥ•ʔ</div><div class="bubble"><i></i><i></i><i></i></div>';
  messageList.appendChild(node);
  messageList.scrollTop = messageList.scrollHeight;
  return node;
}

async function sendMessage(message) {
  const text = message.trim();
  if (!text || sendButton.disabled) return;
  addMessage("user", text);
  messageInput.value = "";
  messageInput.style.height = "auto";
  sendButton.disabled = true;
  const typing = addTyping();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, message: text }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "请求没有成功");
    addMessage("assistant", data.answer, data.tool_calls || [], data.memory_count);
  } catch (error) {
    addMessage("assistant", `哎呀，连接出了点小问题：${error.message}`);
  } finally {
    typing.remove();
    sendButton.disabled = false;
    messageInput.focus();
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(messageInput.value);
});

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

messageInput.addEventListener("input", () => {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 120)}px`;
});

document.querySelectorAll("[data-message]").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.message));
});

clearButton.addEventListener("click", () => {
  messageList.innerHTML = "";
  addMessage("assistant", "好呀，这一页我们重新开始。今天想先做什么？");
});

addMessage("assistant", "你好呀，我是你的小熊日常助手。\n天气、计算、待办，都可以直接告诉我～");
