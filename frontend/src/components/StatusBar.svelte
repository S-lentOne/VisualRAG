<script lang="ts">
    import { status } from "../lib/status";
</script>

<div class="status-bar">
    <div class="status-item">
        <span class="dot {$status.ollama}"></span>
        Ollama
    </div>

    <div class="status-item">
        <span class="dot {$status.camera}"></span>
        Camera
    </div>

    <div class="status-item">
        <span class="dot {$status.yolo}"></span>
        YOLO
    </div>
</div>
<style>
    .status-bar {
        display: flex;
        gap: 1rem;

        padding: 0.75rem 1rem;

        background: var(--panel);
        border-top: 1px solid var(--border);
    }

    .status-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .dot {
        width: 10px;
        height: 10px;

        border-radius: 50%;
    }

    .dot.ready,
    .dot.active {
        background: #22c55e;
    }

    .dot.starting {
        background: #f59e0b;
    }

    .dot.error,
    .dot.offline {
        background: #ef4444;
    }

    .dot.idle,
    .dot.inactive {
        background: #6b7280;
    }
</style>
