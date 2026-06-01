<script lang="ts">
    import { invoke } from "@tauri-apps/api/core";
    import { listen } from "@tauri-apps/api/event";
    import { onMount } from "svelte";
    import { darkMode } from "../lib/theme";
    import { frame } from "../lib/frame";
    import { status } from "../lib/status";
    import "../lib/inputs";

    import CameraFeed from "../components/CameraFeed.svelte";
    import ChatPanel from "../components/ChatPanel.svelte";
    import DetectionPanel from "../components/DetectionPanel.svelte";
    import ScenePanel from "../components/ScenePanel.svelte";
    import TopBar from "../components/TopBar.svelte";
    import SettingsPanel from "../components/SettingsPanel.svelte";
    import StatusBar from "../components/StatusBar.svelte";

    import { detections, scene, chatMessages } from "../lib/stores";

    onMount(async () => {
        await listen("backend-event", (event) => {
            const data = JSON.parse(event.payload as string);

            console.log("BACKEND:", data);

            switch (data.event) {
                case "detections":
                    detections.set(data.data);
                    break;

                case "scene":
                    scene.set(data.data);
                    break;

                case "status":
                    console.log("STATUS:", data.data);
                    break;

                case "error":
                    console.error("BACKEND ERROR:", data.data);
                    break;

                case "frame":
                    frame.set(`data:image/jpeg;base64,${data.data}`);
                    break;

                case "chat_response":
                    chatMessages.update((msgs) => [
                        ...msgs,
                        {
                            sender: "You",
                            text: data.data.question,
                        },
                        {
                            sender: "VisualRAG",
                            text: data.data.answer,
                        },
                    ]);
                    break;
                case "status":
                    status.update((s) => ({
                        ...s,
                        ...data.data,
                    }));
                    break;
            }
        });

        await invoke("start_backend");
    });
</script>

<div class:dark={$darkMode} class:light={!$darkMode}>
    <TopBar />
    <SettingsPanel />

    <div class="layout">
        <div class="left-column">
            <CameraFeed />
            <DetectionPanel />
            <ScenePanel />
        </div>

        <ChatPanel />
    </div>
    <div class="status-bar">
        <StatusBar />
    </div>
</div>

<style>
    :global(body) {
        margin: 0;
        background: #121212;
        color: #ffffff;
        font-family: Inter, sans-serif;
    }

    :global(.dark) {
        --bg: #121212;
        --panel: #1e1e1e;
        --panel-alt: #2a2a2a;

        --text: #ffffff;
        --border: #444;

        --accent: #5865f2;
    }

    :global(.light) {
        --bg: #f5f5f5;
        --panel: #ffffff;
        --panel-alt: #eeeeee;

        --text: #111111;
        --border: #cccccc;

        --accent: #5865f2;
    }

    .layout {
        display: grid;
        grid-template-columns: 3fr 2fr;
        gap: 1rem;

        background: var(--bg);
        color: var(--text);

        height: calc(100vh - 56px);

        padding: 1rem;
        overflow: hidden;
    }
    .left-column {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        overflow-y:auto;
        min-height: 0;
    }
    .layout > * {
        min-height: 0;
    }
    .overlay {
        position: fixed;
        inset: 0;
        z-index: 100;
    }

    .settings-panel {
        position: fixed;
        z-index: 101;
    }
</style>
