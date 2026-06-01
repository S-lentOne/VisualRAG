<script lang="ts">
    import { detections } from "../lib/stores";
</script>

<div class="panel">
    <h2>Detections</h2>

    <div class="chips">
        {#each $detections as detection}
            <span class="chip">{detection}</span>
        {/each}
    </div>
</div>

<style>
    .chips {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }

    .chip {
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        background: var(--panel-alt);
        border: 1px solid var(--border);
        color: var(--text);
    }
    .panel {
        margin-bottom: auto;
        border-radius: 12px;
        margin-top: auto;
        background: var(--panel);
        color: var(--text);
        border: 1px solid var(--border);
        padding: 1rem;
    }
</style>
