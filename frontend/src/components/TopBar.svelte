<script lang="ts">
    import { darkMode } from "../lib/theme";
    import { openUrl } from "@tauri-apps/plugin-opener";
    import { Moon, Sun, Settings } from "lucide-svelte";
    import { settingsOpen } from "../lib/stores";

    function toggleSettings() {
        settingsOpen.update((v) => !v);
    }
    async function openGitHub() {
        await openUrl("https://github.com/your-username/VisualRAG");
    }
    function toggleTheme() {
        darkMode.update((v) => !v);
    }
</script>

<div class="topbar">
    <div class="left">
        <button class="icon-btn" on:click={toggleSettings}>
            <Settings size={20} />
        </button>
    </div>

    <div class="title">VisualRAG</div>

    <div class="right">
        <button on:click={toggleTheme} class="icon-btn">
            {#if $darkMode}
                <Sun size={20} />
            {:else}
                <Moon size={20} />
            {/if}
        </button>

        <button on:click={openGitHub} class="icon-btn">
            <img src="/github.svg" alt="GitHub" />
        </button>
    </div>
</div>

<style>
    .topbar {
        height: 56px;

        display: flex;
        align-items: center;
        justify-content: space-between;

        padding: 0 1rem;

        background: var(--panel);
        color: var(--text);
        border-bottom: 1px solid var(--border);

        position: sticky;
        top: 0;
        z-index: 100;
    }

    .left,
    .right {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 0.5rem;
    }

    .title {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .icon-btn {
        display: flex;
        align-items: center;
        justify-content: center;

        width: 36px;
        height: 36px;

        background: transparent;
        border: none;

        color: var(--text);

        cursor: pointer;
        border-radius: 8px;

        transition: background 0.2s;
    }

    .icon-btn:hover {
        background: var(--panel-alt);
    }
    .icon-btn img {
        width: 25px;
        height: 25px;
    }
</style>
