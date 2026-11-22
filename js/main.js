// Entry point: builds the app and starts it.
import { AIApp } from './app.js';
import { AppModels } from './app-models.js';
import { AppStorage } from './app-storage.js';
import { AppChat } from './app-chat.js';
import { AppTraining } from './app-training.js';
import { AppMemory } from './app-memory.js';
import { AppEval } from './app-eval.js';
import { AppAssistant } from './app-assistant.js';
import { AppQATools } from './app-qa-tools.js';
import { AppScreen } from './app-screen.js';

// Copy every method from each feature class onto AIApp.
for (const Mixin of [AppModels, AppStorage, AppChat, AppTraining, AppMemory, AppEval, AppAssistant, AppQATools, AppScreen]) {
  for (const name of Object.getOwnPropertyNames(Mixin.prototype)) {
    if (name === 'constructor') continue;
    Object.defineProperty(AIApp.prototype, name, Object.getOwnPropertyDescriptor(Mixin.prototype, name));
  }
}

const boot = () => {
  window.app = new AIApp();
  if ('serviceWorker' in navigator && (location.protocol === 'https:' || location.hostname === 'localhost')) {
    navigator.serviceWorker.register('./sw.js').catch(err => console.warn('App shell caching unavailable:', err));
  }
};

// Modules run after the page is parsed, but stay safe either way.
if (document.readyState === 'loading') window.addEventListener('DOMContentLoaded', boot);
else boot();
