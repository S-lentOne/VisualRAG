<script lang="ts">
    import { fly } from "svelte/transition";
    import { invoke } from "@tauri-apps/api/core";

    import { settings } from "../lib/settings";
    import { settingsOpen } from "../lib/stores";

    async function pushSettings() {
        try {
            await invoke("update_settings", {
                settings: $settings,
            });
        } catch (err) {
            console.error("Failed to update settings:", err);
        }
    }
</script>

{#if $settingsOpen}
    <div
        class="overlay"
        on:click={() => settingsOpen.set(false)}
        aria-hidden="true"
    ></div>

    <aside
        class="panel"
        transition:fly={{
            x: -320,
            duration: 250,
        }}
    >
        <h2>Settings</h2>

        <div class="section">
            <label for="camera-index">Camera Index</label>
            <input
                id="camera-index"
                type="number"
                bind:value={$settings.cameraIndex}
                on:change={pushSettings}
            />
        </div>

        <div class="section">
            <label for="detection-interval">
                Detection Interval
            </label>
            <input
                id="detection-interval"
                type="number"
                step="0.1"
                bind:value={$settings.detectionInterval}
                on:change={pushSettings}
            />
        </div>

        <div class="section">
            <label for="model-name">Model</label>
            <input
                id="model-name"
                bind:value={$settings.model}
                on:change={pushSettings}
            />
        </div>

        <div class="section checkbox">
            <label>
                <input
                    type="checkbox"
                    bind:checked={$settings.debug}
                    on:change={pushSettings}
                />
                Debug Mode
            </label>
        </div>
    </aside>
{/if}

<style>
    .overlay {
        position: fixed;
        inset: 0;

        background: rgba(0, 0, 0, 0.25);

        z-index: 900;
    }

    .panel {
        position: fixed;

        top: 56px;
        left: 0;

        width: 320px;
        height: calc(100vh - 56px);

        padding: 1rem;

        background: rgba(30, 30, 30, 0.75);
        backdrop-filter: blur(12px);

        border-right: 1px solid var(--border);

        z-index: 1000;

        overflow-y: auto;

        box-sizing: border-box;
    }

    h2 {
        margin-top: 0;
        margin-bottom: 1.5rem;
    }

    .section {
        display: flex;
        flex-direction: column;

        gap: 0.5rem;

        margin-bottom: 1rem;
    }

    .checkbox {
        flex-direction: row;
    }

    .checkbox label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    input {
        padding: 0.75rem;

        border-radius: 8px;
        border: 1px solid var(--border);

        background: var(--panel-alt);
        color: var(--text);

        box-sizing: border-box;
    }

    input:focus {
        outline: none;
        border-color: var(--accent);
    }
</style>
