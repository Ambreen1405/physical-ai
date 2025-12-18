import { create } from 'zustand';

interface Message {
  text: string;
  sender: 'user' | 'assistant';
}

interface ChatState {
  isChatOpen: boolean;
  messages: Message[];
  isLoading: boolean;
  toggleChat: () => void;
  addMessage: (message: Message) => void;
  clearMessages: () => void;
  setIsLoading: (loading: boolean) => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  isChatOpen: false,
  messages: [],
  isLoading: false,

  toggleChat: () => set((state) => ({ isChatOpen: !state.isChatOpen })),

  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message]
    })),

  clearMessages: () => set({ messages: [] }),

  setIsLoading: (isLoading) => set({ isLoading }),
}));