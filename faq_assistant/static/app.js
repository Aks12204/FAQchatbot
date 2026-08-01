// Minimal & User-Friendly FAQ Assistant JS

document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const chatThread = document.getElementById('chatThread');
    const suggestionsContainer = document.getElementById('suggestions');
    const clearBtn = document.getElementById('clearBtn');

    // Load suggested chips
    loadSuggestions();

    // Event listeners
    chatForm.addEventListener('submit', handleFormSubmit);
    clearBtn.addEventListener('click', clearChat);

    async function loadSuggestions() {
        try {
            const res = await fetch('/api/suggested');
            const data = await res.json();
            if (data.suggested && data.suggested.length > 0) {
                renderChips(data.suggested);
            }
        } catch (err) {
            console.error('Failed to load suggestions:', err);
        }
    }

    function renderChips(chips) {
        suggestionsContainer.innerHTML = '';
        chips.forEach(text => {
            const chip = document.createElement('button');
            chip.className = 'chip';
            chip.textContent = text;
            chip.addEventListener('click', () => {
                userInput.value = text;
                chatForm.dispatchEvent(new Event('submit'));
            });
            suggestionsContainer.appendChild(chip);
        });
    }

    async function handleFormSubmit(e) {
        e.preventDefault();
        const query = userInput.value.trim();
        if (!query) return;

        // 1. Render User Message
        appendMessage(query, 'user');
        userInput.value = '';

        // 2. Show Typing Indicator
        const typingElem = showTypingIndicator();

        try {
            // 3. Send API Request
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: query })
            });

            const data = await response.json();
            
            // Remove typing indicator
            typingElem.remove();

            // 4. Render Bot Answer
            const answer = data.answer || "I'm sorry, I couldn't process your request.";
            const sourceInfo = data.source ? `Source: ${data.source}` : '';
            appendMessage(answer, 'bot', sourceInfo);

        } catch (err) {
            typingElem.remove();
            appendMessage("Unable to reach the server. Please try again later.", 'bot', 'Connection Error');
        }
    }

    function appendMessage(text, sender, metaText = '') {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-message`;

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.textContent = text;
        msgDiv.appendChild(bubble);

        if (metaText && sender === 'bot') {
            const meta = document.createElement('div');
            meta.className = 'meta-badge';
            meta.textContent = metaText;
            msgDiv.appendChild(meta);
        }

        chatThread.appendChild(msgDiv);
        scrollToBottom();
    }

    function showTypingIndicator() {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message bot-message typing-indicator-msg';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.innerHTML = `
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        msgDiv.appendChild(bubble);
        chatThread.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    }

    function clearChat() {
        chatThread.innerHTML = `
            <div class="message system-message">
                <div class="message-bubble">
                    👋 Chat cleared. How can I help you today?
                </div>
            </div>
        `;
    }

    function scrollToBottom() {
        chatThread.scrollTop = chatThread.scrollHeight;
    }
});
