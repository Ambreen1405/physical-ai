import React from 'react';

interface MessageProps {
  text: string;
  sender: 'user' | 'assistant';
}

const ChatMessage: React.FC<MessageProps> = ({ text, sender }) => {
  const isUser = sender === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] rounded-lg px-4 py-2 ${
          isUser
            ? 'bg-primary-600 text-white rounded-tr-none'
            : 'bg-gray-200 text-gray-800 rounded-tl-none'
        }`}
      >
        <div className="whitespace-pre-wrap">{text}</div>
      </div>
    </div>
  );
};

export default ChatMessage;