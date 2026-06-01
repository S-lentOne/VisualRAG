<script lang="ts">
    import { frame } from "../lib/frame";
    import { open } from "@tauri-apps/plugin-dialog";
    import { inputMode } from "../lib/inputs";
    import { invoke } from "@tauri-apps/api/core";

    async function uploadFile() {
        const file = await open({
            multiple: false,
        });

        if (!file || Array.isArray(file)) {
            return;
        }

        const lower = file.toLowerCase();

        if (
            lower.endsWith(".png") ||
            lower.endsWith(".jpg") ||
            lower.endsWith(".jpeg") ||
            lower.endsWith(".webp")
        ) {
            inputMode.set("image");

            await invoke("analyze_image", {
                path: file,
            });
        } else if (
            lower.endsWith(".mp4") ||
            lower.endsWith(".mkv") ||
            lower.endsWith(".avi") ||
            lower.endsWith(".mov")
        ) {
            inputMode.set("video");

            await invoke("analyze_video", {
                path: file,
            });
        }
    }
</script>

<div class="panel">
    <h2>Camera Preview</h2>

    <div class="actions">
        {#if $inputMode !== "none"}
            <button
                on:click={() => {
                    frame.set("");
                    inputMode.set("none");
                }}
            >
                ✕
            </button>
        {/if}

        <button>⛶</button>
    </div>

    {#if $inputMode === "none"}
        <div class="camera-placeholder">Select a source to begin</div>
    {:else}
        {#if $frame}
            <div class="container"><img src={$frame} alt="VisualRAG Output" class="camera-feed" /></div>
        {:else}
            <div class="camera-placeholder">Processing...</div>
        {/if}
    {/if}
    <div class="source-bar">
        <button on:click={() => inputMode.set("live")}> Turn on Webcam </button>

        <button on:click={uploadFile}> Upload File </button>
    </div>
</div>

<style>
    .camera-feed {
        width: auto;
        height: auto;

        object-fit: contain;

        border-radius: 8px;
    }
    .container {
        height: auto;
        min-height: 300px;
        border: 2px var(--border);
        color: var(--text);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .camera-placeholder {
        height: auto;
        min-height: 350px;
        border: 2px dashed var(--border);
        color: var(--text);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .panel {
        background: var(--panel);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
    }
    .source-bar {
        display: flex;
        gap: 0.75rem;

        margin-top: 1rem;
    }

    .source-bar button {
        flex: 1;

        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;

        padding: 0.85rem 1rem;

        background: var(--panel-alt);
        color: var(--text);

        border: 1px solid var(--border);
        border-radius: 12px;

        font-weight: 600;
        font-size: 0.95rem;

        cursor: pointer;

        transition:
            background 0.2s,
            border-color 0.2s,
            transform 0.15s;
    }

    .source-bar button:hover {
        border-color: var(--accent);
        transform: translateY(-1px);
    }

    .source-bar button:active {
        transform: scale(0.98);
    }

    .source-bar button.active {
        background: var(--accent);
        border-color: var(--accent);
        color: white;
    }
</style>
