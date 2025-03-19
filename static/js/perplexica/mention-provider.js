/**
 * @-Mention Provider für das Chat-Interface
 * 
 * Diese Komponente bietet die Funktionalität für das @-Mention-System,
 * einschließlich:
 * - Anzeigen eines Dropdowns, wenn '@' getippt wird
 * - Kategorien und Suchergebnisse anzeigen
 * - Auswahl eines Elements und Einfügen in den Text
 */

import { LitElement, html, css } from 'lit';

export class MentionProvider extends LitElement {
  static properties = {
    isOpen: { type: Boolean },
    query: { type: String },
    categories: { type: Array },
    results: { type: Array },
    loading: { type: Boolean },
    selectedCategory: { type: String },
    selectedIndex: { type: Number },
    position: { type: Object },
    menuWidth: { type: Number },
    textAreaRef: { type: Object },
  };

  static styles = css`
    :host {
      display: block;
      position: relative;
    }

    .mention-dropdown {
      position: fixed;
      background-color: var(--elevated-background, #ffffff);
      border: 1px solid var(--border-color, #e1e4e8);
      border-radius: 6px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
      max-height: 300px;
      overflow-y: auto;
      z-index: 1000;
      width: var(--mention-dropdown-width, 320px);
      font-family: system-ui, sans-serif;
    }

    .mention-header {
      display: flex;
      padding: 8px 12px;
      border-bottom: 1px solid var(--border-color, #e1e4e8);
      font-weight: 600;
      color: var(--text-secondary, #586069);
      font-size: 0.85rem;
    }

    .category-tabs {
      display: flex;
      overflow-x: auto;
      border-bottom: 1px solid var(--border-color, #e1e4e8);
      padding: 0 4px;
    }

    .category-tab {
      padding: 8px 12px;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      white-space: nowrap;
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .category-tab.active {
      border-bottom-color: var(--primary-color, #0366d6);
      color: var(--primary-color, #0366d6);
      font-weight: 600;
    }

    .mention-list {
      padding: 4px 0;
    }

    .mention-item {
      padding: 8px 12px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .mention-item:hover, .mention-item.selected {
      background-color: var(--hover-background, #f6f8fa);
    }

    .item-icon {
      font-size: 1.2rem;
      flex-shrink: 0;
    }

    .item-content {
      flex: 1;
      overflow: hidden;
    }

    .item-title {
      font-weight: 500;
      margin-bottom: 2px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .item-description {
      font-size: 0.85rem;
      color: var(--text-secondary, #586069);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .loading-indicator {
      display: flex;
      justify-content: center;
      padding: 12px;
      color: var(--text-secondary, #586069);
    }

    .no-results {
      padding: 12px;
      color: var(--text-secondary, #586069);
      text-align: center;
    }

    .search-input {
      width: 100%;
      padding: 8px 12px;
      border: none;
      border-bottom: 1px solid var(--border-color, #e1e4e8);
      outline: none;
      font-size: 0.9rem;
    }

    /* Anpassung für dunkles Farbschema */
    @media (prefers-color-scheme: dark) {
      .mention-dropdown {
        --elevated-background: #1e1e1e;
        --border-color: #333;
        --text-secondary: #aaa;
        --hover-background: #252525;
      }
    }
  `;

  constructor() {
    super();
    this.isOpen = false;
    this.query = '';
    this.categories = [];
    this.results = [];
    this.loading = false;
    this.selectedCategory = '';
    this.selectedIndex = -1;
    this.position = { top: 0, left: 0 };
    this.menuWidth = 320;
    this.textAreaRef = null;

    // Kategorien beim Start laden
    this.loadCategories();
  }

  async loadCategories() {
    try {
      const response = await fetch('/models/api/mentions/categories');
      if (response.ok) {
        const data = await response.json();
        this.categories = data.categories;
        if (this.categories.length > 0) {
          this.selectedCategory = this.categories[0].id;
        }
      } else {
        console.error('Fehler beim Laden der Kategorien:', await response.text());
      }
    } catch (error) {
      console.error('Fehler beim Laden der Kategorien:', error);
    }
  }

  async searchMentions(category, query) {
    this.loading = true;
    this.results = [];

    try {
      const response = await fetch(`/models/api/mentions/search?category=${category}&q=${encodeURIComponent(query)}`);
      if (response.ok) {
        const data = await response.json();
        this.results = data.results;
      } else {
        console.error('Fehler bei der Mention-Suche:', await response.text());
      }
    } catch (error) {
      console.error('Fehler bei der Mention-Suche:', error);
    } finally {
      this.loading = false;
    }
  }

  show(textAreaRef, position) {
    this.isOpen = true;
    this.textAreaRef = textAreaRef;
    this.position = position;
    this.selectedIndex = -1;
    this.query = '';
    this.results = [];
    
    // Automatisch erste Kategorie auswählen und Ergebnisse laden
    if (this.categories.length > 0) {
      this.searchMentions(this.selectedCategory, '');
    }
    
    // Dropdown außerhalb des sichtbaren Bereichs vermeiden
    this.updatePosition();
    
    // Event-Listener für Klicks außerhalb des Dropdown-Menüs
    this.setupClickOutsideListener();
  }

  hide() {
    this.isOpen = false;
    this.removeClickOutsideListener();
  }

  setupClickOutsideListener() {
    this.boundClickOutsideHandler = this.handleClickOutside.bind(this);
    document.addEventListener('click', this.boundClickOutsideHandler);
  }

  removeClickOutsideListener() {
    if (this.boundClickOutsideHandler) {
      document.removeEventListener('click', this.boundClickOutsideHandler);
    }
  }

  handleClickOutside(event) {
    if (this.isOpen && !this.contains(event.target)) {
      this.hide();
    }
  }

  updatePosition() {
    // Auf nächsten Render-Zyklus warten, um sicherzustellen, dass das DOM aktualisiert ist
    requestAnimationFrame(() => {
      const dropdown = this.shadowRoot.querySelector('.mention-dropdown');
      if (!dropdown) return;

      // Prüfen, ob das Dropdown unter dem Cursor passt
      const windowHeight = window.innerHeight;
      const dropdownHeight = dropdown.offsetHeight;
      
      // Anpassen der Position, wenn das Dropdown unten nicht passt
      if (this.position.top + dropdownHeight > windowHeight) {
        this.position = {
          ...this.position,
          top: Math.max(windowHeight - dropdownHeight - 10, 10)
        };
      }

      // Aktualisierte Position anwenden
      dropdown.style.top = `${this.position.top}px`;
      dropdown.style.left = `${this.position.left}px`;
      dropdown.style.width = `${this.menuWidth}px`;
    });
  }

  selectCategory(categoryId) {
    this.selectedCategory = categoryId;
    this.selectedIndex = -1;
    this.searchMentions(categoryId, this.query);
  }

  handleInput(event) {
    this.query = event.target.value;
    this.searchMentions(this.selectedCategory, this.query);
  }

  handleKeyDown(event) {
    if (!this.isOpen) return;

    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        this.selectedIndex = Math.min(this.selectedIndex + 1, this.results.length - 1);
        this.scrollToSelectedItem();
        break;
      case 'ArrowUp':
        event.preventDefault();
        this.selectedIndex = Math.max(this.selectedIndex - 1, -1);
        this.scrollToSelectedItem();
        break;
      case 'Enter':
        event.preventDefault();
        if (this.selectedIndex >= 0 && this.results[this.selectedIndex]) {
          this.selectItem(this.results[this.selectedIndex]);
        }
        break;
      case 'Escape':
        event.preventDefault();
        this.hide();
        break;
      case 'Tab':
        event.preventDefault();
        // Wechsle zur nächsten/vorherigen Kategorie
        const categoryIndex = this.categories.findIndex(c => c.id === this.selectedCategory);
        const nextIndex = event.shiftKey 
          ? (categoryIndex - 1 + this.categories.length) % this.categories.length
          : (categoryIndex + 1) % this.categories.length;
        this.selectCategory(this.categories[nextIndex].id);
        break;
    }
  }

  scrollToSelectedItem() {
    requestAnimationFrame(() => {
      const selectedItem = this.shadowRoot.querySelector('.mention-item.selected');
      const mentionList = this.shadowRoot.querySelector('.mention-list');
      
      if (selectedItem && mentionList) {
        const listRect = mentionList.getBoundingClientRect();
        const itemRect = selectedItem.getBoundingClientRect();
        
        if (itemRect.bottom > listRect.bottom) {
          mentionList.scrollTop += (itemRect.bottom - listRect.bottom);
        } else if (itemRect.top < listRect.top) {
          mentionList.scrollTop -= (listRect.top - itemRect.top);
        }
      }
    });
  }

  selectItem(item) {
    // Erstelle eine formatierte Darstellung des ausgewählten Elements
    const formattedMention = `@${this.selectedCategory}:${item.id}`;
    const displayText = item.name || item.title;
    
    // Benutze den TextArea-Ref, um die Auswahl einzufügen
    if (this.textAreaRef) {
      const textArea = this.textAreaRef;
      const textValue = textArea.value;
      const cursorPos = textArea.selectionStart;
      
      // Finde die Position des letzten '@', das vor dem Cursor steht
      const atPos = textValue.substring(0, cursorPos).lastIndexOf('@');
      
      if (atPos !== -1) {
        // Ersetze alles zwischen '@' und Cursor durch die formatierte Mention
        const newValue = 
          textValue.substring(0, atPos) + 
          `@${displayText}` + 
          textValue.substring(cursorPos);
        
        // Aktualisiere den Wert des TextArea
        textArea.value = newValue;
        
        // Setze den Cursor hinter die eingefügte Mention
        const newCursorPos = atPos + displayText.length + 1;
        textArea.setSelectionRange(newCursorPos, newCursorPos);
        
        // Simuliere ein Input-Event, um andere Komponenten zu informieren
        textArea.dispatchEvent(new Event('input', { bubbles: true }));
        
        // Speichere die Mention-ID im Datensatz der TextArea
        if (!textArea.dataset.mentions) {
          textArea.dataset.mentions = JSON.stringify({});
        }
        
        const mentionsData = JSON.parse(textArea.dataset.mentions);
        mentionsData[displayText] = {
          type: this.selectedCategory,
          id: item.id,
          original: formattedMention
        };
        
        textArea.dataset.mentions = JSON.stringify(mentionsData);
        
        // Ereignis auslösen für andere Komponenten
        const event = new CustomEvent('mention-selected', {
          detail: {
            mention: {
              type: this.selectedCategory,
              id: item.id,
              text: displayText,
              data: item
            }
          },
          bubbles: true,
          composed: true
        });
        
        this.dispatchEvent(event);
      }
      
      // Schließe das Dropdown-Menü
      this.hide();
      
      // Fokus zurück auf TextArea
      textArea.focus();
    }
  }

  render() {
    if (!this.isOpen) {
      return html``;
    }

    return html`
      <div 
        class="mention-dropdown" 
        @keydown="${this.handleKeyDown}" 
        style="top: ${this.position.top}px; left: ${this.position.left}px; width: ${this.menuWidth}px;">
        
        <div class="mention-header">
          Erwähne Dokumente oder Projekte
        </div>
        
        <input
          type="text"
          class="search-input"
          placeholder="Suchen..."
          .value="${this.query}"
          @input="${this.handleInput}"
          autofocus
        />
        
        <div class="category-tabs">
          ${this.categories.map(category => html`
            <div 
              class="category-tab ${category.id === this.selectedCategory ? 'active' : ''}" 
              @click="${() => this.selectCategory(category.id)}">
              <span class="category-icon">${category.icon}</span>
              <span>${category.name}</span>
            </div>
          `)}
        </div>
        
        <div class="mention-list">
          ${this.loading 
            ? html`<div class="loading-indicator">Wird geladen...</div>` 
            : this.results.length === 0 
              ? html`<div class="no-results">Keine Ergebnisse gefunden</div>`
              : this.results.map((item, index) => html`
                <div 
                  class="mention-item ${index === this.selectedIndex ? 'selected' : ''}" 
                  @click="${() => this.selectItem(item)}"
                  @mouseenter="${() => this.selectedIndex = index}">
                  <div class="item-icon">${item.icon || this.categories.find(c => c.id === this.selectedCategory)?.icon || '📄'}</div>
                  <div class="item-content">
                    <div class="item-title">${item.name || item.title}</div>
                    <div class="item-description">${item.description || item.subtitle || ''}</div>
                  </div>
                </div>
              `)
          }
        </div>
      </div>
    `;
  }
}

customElements.define('mention-provider', MentionProvider); 