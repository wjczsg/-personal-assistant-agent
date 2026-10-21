const messageList = document.querySelector("#messageList");
const chatForm = document.querySelector("#chatForm");
const messageInput = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const clearButton = document.querySelector("#clearButton");
const documentFile = document.querySelector("#documentFile");
const attachmentButton = document.querySelector("#attachmentButton");
const attachmentPreview = document.querySelector("#attachmentPreview");
const attachmentStatus = document.querySelector("#attachmentStatus");
const userId = localStorage.getItem("assistant-user-id") || "web-user";

attachmentButton.addEventListener("click", () => documentFile.click());

function validAttachment(file) {
  return /\.(txt|md)$/i.test(file.name) && file.size <= 1024 * 1024;
}

function renderAttachmentPreview() {
  attachmentPreview.replaceChildren();
  const file = documentFile.files[0];
  if (!file) {
    attachmentPreview.hidden = true;
    return;
  }
  if (!validAttachment(file)) {
    documentFile.value = "";
    attachmentPreview.hidden = true;
    attachmentStatus.textContent = "请选择不超过 1 MB 的 .txt 或 .md 文件。";
    return;
  }
  attachmentStatus.textContent = "附件会与这条消息一起发送。";
  const chip = document.createElement("span");
  chip.className = "attachment-chip";
  chip.textContent = `📄 ${file.name}`;
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "attachment-remove";
  remove.setAttribute("aria-label", "移除附件");
  remove.textContent = "×";
  remove.addEventListener("click", () => {
    documentFile.value = "";
    renderAttachmentPreview();
  });
  chip.appendChild(remove);
  attachmentPreview.appendChild(chip);
  attachmentPreview.hidden = false;
}

documentFile.addEventListener("change", renderAttachmentPreview);

function addMessage(role, text, toolCalls = [], memoryCount = null, attachments = []) {
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
  if (role === "user") {
    attachments.forEach((attachment) => {
      const badge = document.createElement("div");
      badge.className = "message-attachment";
      badge.textContent = `📄 ${attachment.filename} · #${attachment.document_id}${attachment.indexed ? "" : " · 索引失败"}`;
      content.appendChild(badge);
    });
  }

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

async function sendMessage(message, includeAttachment = true) {
  const file = includeAttachment ? documentFile.files[0] : null;
  const text = message.trim() || (file ? "请介绍这份文档" : "");
  if (!text || sendButton.disabled) return;
  if (file && !validAttachment(file)) {
    attachmentStatus.textContent = "请选择不超过 1 MB 的 .txt 或 .md 文件。";
    return;
  }
  sendButton.disabled = true;
  attachmentButton.disabled = true;
  const typing = addTyping();

  try {
    let response;
    if (file) {
      const formData = new FormData();
      formData.append("user_id", userId);
      formData.append("message", text);
      formData.append("file", file);
      response = await fetch("/api/chat/with-attachment", { method: "POST", body: formData });
    } else {
      response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, message: text }),
      });
    }
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "请求没有成功");
    addMessage("user", text, [], null, data.attachments || []);
    addMessage("assistant", data.answer, data.tool_calls || [], data.memory_count);
    messageInput.value = "";
    messageInput.style.height = "auto";
    if (file) {
      documentFile.value = "";
      renderAttachmentPreview();
    }
    attachmentStatus.textContent = "Enter 发送 · Shift + Enter 换行 · 附件支持 UTF-8 .txt / .md，最大 1 MB";
  } catch (error) {
    addMessage("assistant", `哎呀，连接出了点小问题：${error.message}`);
  } finally {
    typing.remove();
    sendButton.disabled = false;
    attachmentButton.disabled = false;
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
  button.addEventListener("click", () => sendMessage(button.dataset.message, false));
});

clearButton.addEventListener("click", () => {
  messageList.innerHTML = "";
  addMessage("assistant", "好呀，这一页我们重新开始。今天想先做什么？");
});

addMessage("assistant", "你好呀，我是你的小熊日常助手。\n天气、计算、待办，都可以直接告诉我～");
