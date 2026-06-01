<script lang="ts">
    import { invoke } from "@tauri-apps/api/core";
    import { chatMessages } from "../lib/stores";

    let message = "";

    async function sendMessage() {
        const text = message.trim();

        if (!text) {
            return;
        }

        await invoke("send_chat", {
            message: text,
        });

        message = "";
    }
</script>

<div class="panel">
    <h2>Chat</h2>
    <div class="chat-history">
        {#each $chatMessages as msg}
            <div class="message">
                <strong>{msg.sender}</strong>
                <p>{msg.text}</p>
            </div>
        {/each}
    </div>
    <div class="input-row">
        <input
            bind:value={message}
            placeholder="Ask about the scene..."
            on:keydown={(e) => {
                if (e.key === "Enter") {
                    sendMessage();
                }
            }}
        />

        <button on:click={sendMessage}> Send </button>
    </div>
</div>

<style>
    .panel {
        flex: 1;

        display: flex;
        flex-direction: column;
        background: var(--panel);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        min-height:0;
    }
    .chat-history {
        flex: 1;
        overflow-y: auto;

        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        min-height:0;
        margin-bottom: 1rem;
    }
    .message {
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
    }
    .message p {
        margin: 0.3rem 0 0;
    }
    .input-row {
        display: flex;
        gap: 0.75rem;

        margin-top: auto;
    }
    input {
        flex: 1;

        color: white;

        background: var(--panel-alt);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 10px;

        padding: 0.9rem 1rem;

        font-size: 0.95rem;
    }

    input:focus {
        outline: none;
        border-color: #5865f2;
    }

    button {
        background: #5865f2;
        color: white;

        border: none;
        border-radius: 10px;
        background: var(--accent);
        color: white;
        padding: 0.9rem 1.2rem;

        font-weight: 600;

        cursor: pointer;

        transition: 0.2s;
    }

    button:hover {
        background: #6875ff;
    }

    button:active {
        transform: scale(0.98);
    }

    /*Shadow and border*/
    .panel {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .message {
        border-left: 3px solid #5865f2;
    }
</style>
