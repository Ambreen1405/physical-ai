import React, { useState, useRef, useEffect } from 'react';
import { useChatStore } from './chatStore';
import ChatMessage from './ChatMessage';

const ChatPanel = () => {
  const { isChatOpen, toggleChat, messages, addMessage, isLoading } = useChatStore();
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim() === '') return;

    // Add user message
    addMessage({ text: inputValue, sender: 'user' });
    const userMessage = inputValue;
    setInputValue('');

    // Simulate AI response (in a real implementation, this would call an API)
    setTimeout(() => {
      const response = getAIResponse(userMessage);
      addMessage({ text: response, sender: 'assistant' });
    }, 500);
  };

  const getAIResponse = (message: string): string => {
    const lowerMsg = message.toLowerCase();

    if (lowerMsg.includes('hello') || lowerMsg.includes('hi') || lowerMsg.includes('hey')) {
      return "Hello! I'm your AI-powered textbook assistant. I can help you with questions about the current chapter or related robotics concepts. What would you like to know?";
    } else if (lowerMsg.includes('ros') || lowerMsg.includes('robot operating system')) {
      return "ROS (Robot Operating System) is a flexible framework for writing robot software. It provides services like hardware abstraction, device drivers, libraries, visualizers, message-passing, and package management. In ROS 2, communication is based on DDS (Data Distribution Service) for improved real-time performance and security.";
    } else if (lowerMsg.includes('urdf') || lowerMsg.includes('unified robot')) {
      return "URDF (Unified Robot Description Format) is an XML format for representing a robot model. It defines the physical and visual properties of a robot, including links (rigid bodies), joints (connections between links), visual and collision properties, inertial properties, and materials. URDF is essential for robot simulation and visualization.";
    } else if (lowerMsg.includes('topic') || lowerMsg.includes('publish') || lowerMsg.includes('subscribe')) {
      return "In ROS, topics enable asynchronous, many-to-many communication using the publisher-subscriber pattern. Publishers send messages to topics, and subscribers receive messages from topics. This decouples the sender and receiver in time and space, allowing for flexible and robust robot architectures.";
    } else if (lowerMsg.includes('service') || lowerMsg.includes('request') || lowerMsg.includes('response')) {
      return "Services in ROS enable synchronous, one-to-one communication where a client sends a request and waits for a response from a server. This is useful for operations that require a specific response, like configuration changes or action triggering.";
    } else if (lowerMsg.includes('module') && (lowerMsg.includes('1') || lowerMsg.includes('one'))) {
      return "Module 1 covers ROS 2 fundamentals, including architecture, nodes, topics, services, Python development with rclpy, URDF for robot description, and launch files. It provides the foundational knowledge needed for robotics development.";
    } else if (lowerMsg.includes('thank')) {
      return "You're welcome! Is there anything else I can help you with regarding the textbook content?";
    } else if (lowerMsg.includes('help')) {
      return "I can help you understand concepts from the Physical AI & Humanoid Robotics textbook. You can ask me about: ROS 2 architecture, nodes and communication patterns, Python development with rclpy, URDF robot modeling, launch files, simulation environments, NVIDIA Isaac, or any other topic covered in the textbook.";
    } else {
      return "I'm your AI-powered textbook assistant. I can help explain concepts from the Physical AI & Humanoid Robotics textbook. For example, you can ask me about ROS 2, URDF, simulation environments, or specific chapters from the textbook. Could you rephrase your question or ask about a specific topic?";
    }
  };

  if (!isChatOpen) return null;

  return (
    <div className="fixed bottom-20 right-6 w-96 h-[500px] bg-white rounded-lg shadow-xl border border-gray-200 flex flex-col z-50">
      <div className="bg-primary-600 text-white p-4 rounded-t-lg flex justify-between items-center">
        <h3 className="font-bold">Textbook Assistant</h3>
        <button
          onClick={toggleChat}
          className="text-white hover:text-gray-200 focus:outline-none"
          aria-label="Close chat"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 bg-gray-50">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-gray-500">
            <p>Hello! I'm your AI-powered textbook assistant.</p>
            <p>Ask me anything about the current chapter or related robotics concepts.</p>
          </div>
        ) : (
          <div className="space-y-3">
            {messages.map((msg, index) => (
              <ChatMessage key={index} text={msg.text} sender={msg.sender} />
            ))}
            {isLoading && (
              <div className="flex items-center space-x-2">
                <div className="bg-primary-100 rounded-full p-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-primary-600 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-primary-600 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    <div className="w-2 h-2 bg-primary-600 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="p-4 border-t border-gray-200 bg-white">
        <div className="flex space-x-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask a question about the textbook..."
            className="flex-1 border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            aria-label="Type your message"
          />
          <button
            type="submit"
            disabled={inputValue.trim() === ''}
            className={`px-4 py-2 rounded ${
              inputValue.trim() === ''
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-primary-600 text-white hover:bg-primary-700'
            }`}
            aria-label="Send message"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatPanel;