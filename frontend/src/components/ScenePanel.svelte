<script lang="ts">
    import { scene } from "../lib/stores";
</script>

<div class="panel">
    <h2>Scene Analysis</h2>

    <div class="analysis">
        {$scene}
    </div>
</div>

<style>
    .panel {
        border-radius: 12px;
        padding: 1rem;
        margin-top: auto;
        background: var(--panel);
        color: var(--text);
        border: 1px solid var(--border);
    }

    .analysis {
        line-height: 1.6;
        padding: 1rem;
        border-radius: 8px;
        background: var(--panel-alt);
        min-height: 200px;
        height: auto;
        overflow-y: auto;
    }
</style>
