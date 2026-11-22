// Save status and backup safety net: "last saved" indicator, export-everything reminder,
// persistent-storage request. Methods here are copied onto AIApp in main.js, so `this` is the app.

const LS_LAST_BACKUP = 'teachable_ai_last_backup_v1';
const LS_FIRST_SEEN = 'teachable_ai_first_seen_v1';
const LS_SNOOZE_UNTIL = 'teachable_ai_backup_snooze_v1';
const DAY = 24 * 60 * 60 * 1000;

export class AppBackup {
  initBackupUI() {
    this.saveState = 'idle'; // idle | saving | saved | failed
    this.storagePersisted = null;
    this.storageUsageBytes = null;
    this.lastBackupAt = Number(this.lsRead(LS_LAST_BACKUP)) || null;
    if (!this.lsRead(LS_FIRST_SEEN)) this.lsWrite(LS_FIRST_SEEN, String(Date.now()));

    document.getElementById('save-status-btn')?.addEventListener('click', () => this.exportAllModelsBundle());
    document.getElementById('backup-export-btn')?.addEventListener('click', () => this.exportAllModelsBundle());
    document.getElementById('backup-later-btn')?.addEventListener('click', () => this.snoozeBackupReminder());
    document.getElementById('backup-dismiss-btn')?.addEventListener('click', () => this.snoozeBackupReminder());

    // Keep "saved 5 min ago" fresh, and refresh when the tab comes back to the front.
    setInterval(() => this.updateSaveIndicator(), 30000);
    document.addEventListener('visibilitychange', () => { if (!document.hidden) this.updateSaveIndicator(); });

    if (navigator.storage && navigator.storage.persisted) {
      navigator.storage.persisted().then(p => { this.storagePersisted = p; this.updateSaveIndicator(); }).catch(() => {});
    }
    this.updateSaveIndicator();
  }

  lsRead(key) { try { return localStorage.getItem(key); } catch (e) { return null; } }

  lsWrite(key, value) { try { localStorage.setItem(key, value); } catch (e) { /* storage may be blocked */ } }

  /* ----- state changes reported by app-storage.js ----- */

  setSaveState(state) {
    this.saveState = state;
    this.updateSaveIndicator();
  }

  markSaved(timestamp, backend) {
    this.lastSavedAt = timestamp;
    this.storageBackend = backend;
    this.saveState = 'saved';
    this.refreshStorageUsage();
    this.updateSaveIndicator();
  }

  markSaveFailed() {
    this.saveState = 'failed';
    this.updateSaveIndicator();
  }

  markBackedUp() {
    this.lastBackupAt = Date.now();
    this.lsWrite(LS_LAST_BACKUP, String(this.lastBackupAt));
    this.lsWrite(LS_SNOOZE_UNTIL, '0');
    this.updateSaveIndicator();
  }

  snoozeBackupReminder() {
    this.lsWrite(LS_SNOOZE_UNTIL, String(Date.now() + DAY));
    this.updateSaveIndicator();
  }

  // Called after the user finishes a training run: the best moment to ask for a first backup.
  nudgeBackup() {
    this.requestPersistentStorage();
    this.updateSaveIndicator(!this.lastBackupAt);
  }

  async requestPersistentStorage() {
    if (!navigator.storage || !navigator.storage.persist || this.storagePersisted) return;
    try {
      this.storagePersisted = await navigator.storage.persist();
      this.updateSaveIndicator();
    } catch (e) { /* not supported or declined */ }
  }

  async refreshStorageUsage() {
    if (!navigator.storage || !navigator.storage.estimate) return;
    try {
      const estimate = await navigator.storage.estimate();
      this.storageUsageBytes = estimate.usage || 0;
      this.updateSaveIndicator();
    } catch (e) { /* ignore */ }
  }

  /* ----- display ----- */

  formatAgo(timestamp) {
    if (!timestamp) return 'never';
    const diff = Math.max(0, Date.now() - timestamp);
    const min = Math.floor(diff / 60000);
    if (min < 1) return 'just now';
    if (min < 60) return min + ' min ago';
    const hours = Math.floor(min / 60);
    if (hours < 24) return hours + ' h ago';
    const days = Math.floor(hours / 24);
    return days === 1 ? 'yesterday' : days + ' days ago';
  }

  // Short form for the header pill, e.g. "now", "5m", "2h", "3d".
  formatAgoShort(timestamp) {
    const min = Math.floor(Math.max(0, Date.now() - timestamp) / 60000);
    if (min < 1) return 'now';
    if (min < 60) return min + 'm';
    if (min < 1440) return Math.floor(min / 60) + 'h';
    return Math.floor(min / 1440) + 'd';
  }

  formatBytes(bytes) {
    if (bytes === null || bytes === undefined) return '';
    if (bytes < 1024 * 1024) return Math.max(1, Math.round(bytes / 1024)) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  isBackupDue(force = false) {
    const hasWork = this.models.some(m => m.engine && m.engine.isTrained);
    if (!hasWork || !this.lastSavedAt) return false;
    if (Number(this.lsRead(LS_SNOOZE_UNTIL)) > Date.now()) return false;
    if (this.lastBackupAt && this.lastSavedAt <= this.lastBackupAt) return false; // nothing new since the last backup
    if (force) return true;
    const reference = this.lastBackupAt || Number(this.lsRead(LS_FIRST_SEEN)) || Date.now();
    return Date.now() - reference > (this.lastBackupAt ? 7 * DAY : DAY);
  }

  updateSaveIndicator(forceReminder = false) {
    const dot = document.getElementById('save-status-dot');
    const text = document.getElementById('save-status-text');
    const button = document.getElementById('save-status-btn');
    if (forceReminder) this._reminderForced = true;
    const due = this.isBackupDue(this._reminderForced);
    if (!due) this._reminderForced = false;

    const colors = { idle: 'bg-slate-500', saving: 'bg-amber-400 animate-pulse', saved: 'bg-emerald-500', failed: 'bg-rose-500' };
    let label;
    if (this.saveState === 'saving') label = 'Saving…';
    else if (this.saveState === 'failed') label = 'Not saved!';
    else if (this.lastSavedAt) label = 'Saved ' + this.formatAgoShort(this.lastSavedAt);
    else label = 'Unsaved';

    if (dot) dot.className = 'w-2 h-2 rounded-full shrink-0 ' + (colors[this.saveState] || colors.idle);
    if (text) text.textContent = label;

    const persistText = this.storagePersisted === true ? 'protected from automatic clean-up' : (this.storagePersisted === false ? 'the browser may clear it when space runs low' : 'persistence unknown');
    const details = [
      this.lastSavedAt ? 'Last saved: ' + new Date(this.lastSavedAt).toLocaleString() : 'Nothing saved yet',
      this.storageBackend ? 'Stored in ' + this.storageBackend : '',
      'Last backup: ' + (this.lastBackupAt ? new Date(this.lastBackupAt).toLocaleString() : 'never'),
      'Browser storage: ' + persistText
    ].filter(Boolean);
    if (button) {
      button.title = details.join(' · ') + ' — tap to export everything';
      button.classList.toggle('ring-1', due);
      button.classList.toggle('ring-amber-400/60', due);
    }

    const bar = document.getElementById('backup-reminder');
    if (bar) {
      bar.classList.toggle('hidden', !due);
      const msg = document.getElementById('backup-reminder-text');
      if (msg && due) {
        msg.textContent = this.lastBackupAt
          ? `You have changes that are not in a backup yet (last backup: ${this.formatAgo(this.lastBackupAt)}). Your models live only in this browser, and phones can clear browser storage.`
          : 'You have not backed up your models yet. They live only in this browser, and phones can clear browser storage without warning.';
      }
    }

    const line = document.getElementById('storage-status-line');
    if (line) {
      const usage = this.storageUsageBytes !== null ? ` · ${this.formatBytes(this.storageUsageBytes)} used` : '';
      line.textContent = `Last saved: ${this.lastSavedAt ? this.formatAgo(this.lastSavedAt) : 'not yet'} · Last backup: ${this.lastBackupAt ? this.formatAgo(this.lastBackupAt) : 'never'} · ${this.storageBackend || 'storage not used yet'}${usage}`;
    }
  }
}
