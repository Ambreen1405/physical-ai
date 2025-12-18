import React from 'react';
import ChatAssistant from './index';

// Root component that wraps the entire app
const Root: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <>
      {children}
      <ChatAssistant />
    </>
  );
};

export default Root;