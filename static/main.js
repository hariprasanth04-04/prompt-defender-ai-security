/**
 * PromptDefender Pro — main.js
 * Global utilities and initializations
 */

window.PromptDefender = {
    version: '2.1',
    startTime: new Date(),

    // Unified API call
    async apiCall(endpoint, options = {}) {
        try {
            const res = await fetch(endpoint, {
                headers: { 'Content-Type': 'application/json', ...options.headers },
                ...options
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return await res.json();
        } catch (err) {
            console.error('[PromptDefender] API error:', err);
            throw err;
        }
    },

    // Format timestamp
    formatDate(dateString) {
        return new Date(dateString).toLocaleString();
    },

    // Format confidence
    formatConfidence(c) {
        return (c * 100).toFixed(1) + '%';
    },

    // Export logs to CSV
    async exportLogs() {
        try {
            const data = await this.apiCall('/api/logs');
            if (!data.logs || data.logs.length === 0) {
                showNotification('No logs to export', 'warning');
                return;
            }
            let csv = 'Timestamp,Prompt,Status,Layer,Reason,Confidence\n';
            data.logs.forEach(l => {
                csv += [
                    `"${l.timestamp}"`,
                    `"${l.prompt.replace(/"/g, '""')}"`,
                    `"${l.blocked ? 'BLOCKED' : 'ALLOWED'}"`,
                    `"${l.layer || ''}"`,
                    `"${(l.reason || '').replace(/"/g, '""')}"`,
                    `"${(l.ml_confidence * 100).toFixed(1)}%"`
                ].join(',') + '\n';
            });
            const blob = new Blob([csv], { type: 'text/csv' });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = `pd-logs-${new Date().toISOString().split('T')[0]}.csv`;
            a.click();
            URL.revokeObjectURL(a.href);
        } catch(e) {
            console.error('Export failed:', e);
        }
    },

    // Uptime
    getUptime() {
        const ms = Date.now() - this.startTime;
        const s = Math.floor(ms / 1000);
        const m = Math.floor(s / 60);
        const h = Math.floor(m / 60);
        if (h > 0) return `${h}h ${m % 60}m`;
        if (m > 0) return `${m}m ${s % 60}s`;
        return `${s}s`;
    }
};

// Keyboard shortcut: Ctrl+/ focuses search
document.addEventListener('keydown', e => {
    if (e.ctrlKey && e.key === '/') {
        e.preventDefault();
        const search = document.querySelector('.topbar-search input');
        if (search) search.focus();
    }
    if (e.key === 'Escape') {
        const active = document.activeElement;
        if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')) {
            active.blur();
        }
    }
});
