import React, { useState } from 'react';
import { useChatStore } from './chatStore';

const ChatButton = () => {
  const { toggleChat, isChatOpen } = useChatStore();

  return (
    <button
      className="chat-assistant-button"
      onClick={toggleChat}
      aria-label={isChatOpen ? "Close chat assistant" : "Open chat assistant"}
    >
      <svg
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="M21 15C21 15.5304 20.7893 16.0391 20.4142 16.4142C20.0391 16.7893 19.5304 17 19 17H17L14.25 20.75C14.1755 20.8506 14.079 20.9337 13.9676 20.9939C13.8562 21.0541 13.733 21.0896 13.606 21.098C13.479 21.1065 13.3518 21.0877 13.2328 21.0429C13.1137 20.998 13.0063 20.9283 12.918 20.838L8.5 16.5H5C4.46957 16.5 3.96086 16.2893 3.58579 15.9142C3.21071 15.5391 3 15.0304 3 14.5V5C3 4.46957 3.21071 3.96086 3.58579 3.58579C3.96086 3.21071 4.46957 3 5 3H19C19.5304 3 20.0391 3.21071 20.4142 3.58579C20.7893 3.96086 21 4.46957 21 5V15Z"
          stroke="white"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </button>
  );
};

export default ChatButton;