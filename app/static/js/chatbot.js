const ngrokUrl = "https://467c-34-19-11-104.ngrok-free.app"; // Replace with your actual ngrok URL

chatForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const query = userInput.value.trim();
    if (!query) return;
    
    // Display user message
    const userMessage = document.createElement('div');
    userMessage.classList.add('item', 'right');
    userMessage.innerHTML = `<div class="msg"><p>${query}</p></div>`;
    chatBox.appendChild(userMessage);
    
    // Reset input field
    userInput.value = '';

    // Fetch response from Flask backend running on Google Colab via ngrok
    try {
        const response = await fetch(`${ngrokUrl}/get_response`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        });
        
        const data = await response.json();
        const answer = data.response;

        // Display chatbot response
        const botMessage = document.createElement('div');
        botMessage.classList.add('item');
        botMessage.innerHTML = `
            <div class="icon"><i class="fa fa-user"></i></div>
            <div class="msg"><p>${answer}</p></div>
        `;
        chatBox.appendChild(botMessage);
        
        // Scroll chat to the bottom
        chatBox.scrollTop = chatBox.scrollHeight;

    } catch (error) {
        console.error('Error fetching response:', error);
    }
});