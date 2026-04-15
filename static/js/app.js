/**
 * AutoAttendance App Controller
 * Core application functionality: modals, dropdowns, forms, notifications
 */

(function() {
  'use strict';

  // ======================================================================
  // MODAL SYSTEM
  // ======================================================================
  
  const Modal = {
    open: function(modalId) {
      const overlay = document.getElementById(modalId);
      if (!overlay) return;
      overlay.classList.add('active');
      document.body.style.overflow = 'hidden';
      
      // Focus first focusable element
      const firstFocusable = overlay.querySelector('button, [href], input, select, textarea');
      if (firstFocusable) firstFocusable.focus();
    },

    close: function(modalId) {
      const overlay = document.getElementById(modalId);
      if (!overlay) return;
      overlay.classList.remove('active');
      document.body.style.overflow = '';
    },

    toggle: function(modalId) {
      const overlay = document.getElementById(modalId);
      if (!overlay) return;
      overlay.classList.contains('active') ? this.close(modalId) : this.open(modalId);
    }
  };

  // Auto-bind modal close buttons
  document.addEventListener('click', function(e) {
    if (e.target.classList.contains('modal-close')) {
      const overlay = e.target.closest('.modal-overlay');
      if (overlay) {
        const modalId = overlay.id;
        Modal.close(modalId);
      }
    }
  });

  // Close modal on ESC key
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      const activeModal = document.querySelector('.modal-overlay.active');
      if (activeModal) Modal.close(activeModal.id);
    }
  });

  // Close modal when clicking outside
  document.addEventListener('click', function(e) {
    if (e.target.classList.contains('modal-overlay')) {
      Modal.close(e.target.id);
    }
  });

  window.Modal = Modal;

  // ======================================================================
  // DROPDOWN SYSTEM
  // ======================================================================

  const Dropdown = {
    open: function(triggerId) {
      const trigger = document.getElementById(triggerId);
      if (!trigger) return;
      
      const menuId = trigger.getAttribute('data-menu-id');
      const menu = document.getElementById(menuId);
      if (!menu) return;
      
      menu.classList.add('active');
      trigger.setAttribute('aria-expanded', 'true');
    },

    close: function(triggerId) {
      const trigger = document.getElementById(triggerId);
      if (!trigger) return;
      
      const menuId = trigger.getAttribute('data-menu-id');
      const menu = document.getElementById(menuId);
      if (!menu) return;
      
      menu.classList.remove('active');
      trigger.setAttribute('aria-expanded', 'false');
    },

    toggle: function(triggerId) {
      const trigger = document.getElementById(triggerId);
      if (!trigger) return;
      
      const menuId = trigger.getAttribute('data-menu-id');
      const menu = document.getElementById(menuId);
      if (!menu) return;
      
      menu.classList.contains('active') ? this.close(triggerId) : this.open(triggerId);
    }
  };

  // Auto-bind dropdowns
  document.addEventListener('click', function(e) {
    // Dropdown trigger
    if (e.target.closest('[data-menu-id]')) {
      const trigger = e.target.closest('[data-menu-id]');
      const triggerId = trigger.id;
      if (triggerId) Dropdown.toggle(triggerId);
      e.stopPropagation();
    }

    // Dropdown item selection
    if (e.target.closest('.dropdown-item')) {
      const item = e.target.closest('.dropdown-item');
      const value = item.getAttribute('data-value') || item.textContent.trim();
      
      // Dispatch custom event
      const event = new CustomEvent('dropdown-select', {
        detail: { value, item }
      });
      document.dispatchEvent(event);
    }

    // Close dropdowns when clicking outside
    const openDropdown = document.querySelector('.dropdown-menu.active');
    if (openDropdown && !e.target.closest('.dropdown-container')) {
      const trigger = document.querySelector('[data-menu-id="' + openDropdown.id + '"]');
      if (trigger) Dropdown.close(trigger.id);
    }
  });

  // Close dropdown on ESC key
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      const openDropdown = document.querySelector('.dropdown-menu.active');
      if (openDropdown) {
        const trigger = document.querySelector('[data-menu-id="' + openDropdown.id + '"]');
        if (trigger) Dropdown.close(trigger.id);
      }
    }
  });

  window.Dropdown = Dropdown;

  // ======================================================================
  // TOAST NOTIFICATIONS
  // ======================================================================

  const Toast = {
    show: function(message, type = 'info', duration = 5000) {
      const container = document.querySelector('.toast-container') || this._createContainer();
      
      const toast = document.createElement('div');
      toast.className = `toast ${type}`;
      
      const icons = {
        success: 'check_circle',
        error: 'error',
        warning: 'warning',
        info: 'info'
      };

      const titles = {
        success: 'Success',
        error: 'Error',
        warning: 'Warning',
        info: 'Info'
      };

      toast.innerHTML = `
        <span class="toast-icon material-symbols-outlined">${icons[type]}</span>
        <div class="toast-content">
          <div class="toast-title">${titles[type]}</div>
          <div class="toast-message">${message}</div>
        </div>
        <button class="toast-close material-symbols-outlined" aria-label="Dismiss">close</button>
      `;

      container.appendChild(toast);

      // Close button listener
      toast.querySelector('.toast-close').addEventListener('click', () => {
        toast.remove();
      });

      // Auto-dismiss
      if (duration > 0) {
        setTimeout(() => {
          toast.style.animation = 'fadeOut var(--transition-base)';
          setTimeout(() => toast.remove(), 250);
        }, duration);
      }

      return toast;
    },

    success: function(message, duration = 5000) {
      return this.show(message, 'success', duration);
    },

    error: function(message, duration = 5000) {
      return this.show(message, 'error', duration);
    },

    warning: function(message, duration = 5000) {
      return this.show(message, 'warning', duration);
    },

    info: function(message, duration = 5000) {
      return this.show(message, 'info', duration);
    },

    _createContainer: function() {
      const container = document.createElement('div');
      container.className = 'toast-container';
      document.body.appendChild(container);
      return container;
    }
  };

  window.Toast = Toast;

  // ======================================================================
  // FORM UTILITIES
  // ======================================================================

  const Form = {
    validate: function(formElement) {
      if (!formElement) return true;
      return formElement.checkValidity();
    },

    getFormData: function(formElement) {
      if (!formElement) return null;
      return new FormData(formElement);
    },

    serialize: function(formElement) {
      if (!formElement) return {};
      const data = new FormData(formElement);
      const obj = {};
      for (let [key, value] of data.entries()) {
        obj[key] = value;
      }
      return obj;
    },

    reset: function(formElement) {
      if (!formElement) return;
      formElement.reset();
    },

    disable: function(formElement) {
      if (!formElement) return;
      const inputs = formElement.querySelectorAll('input, textarea, select, button');
      inputs.forEach(input => input.disabled = true);
    },

    enable: function(formElement) {
      if (!formElement) return;
      const inputs = formElement.querySelectorAll('input, textarea, select, button');
      inputs.forEach(input => input.disabled = false);
    }
  };

  window.Form = Form;

  // ======================================================================
  // TABLE UTILITIES
  // ======================================================================

  const Table = {
    sortColumn: function(tableElement, columnIndex, ascending = true) {
      if (!tableElement) return;
      
      const rows = Array.from(tableElement.querySelectorAll('tbody tr'));
      rows.sort((a, b) => {
        const cellA = a.cells[columnIndex].textContent.trim();
        const cellB = b.cells[columnIndex].textContent.trim();
        
        // Try numeric comparison first
        const numA = parseFloat(cellA);
        const numB = parseFloat(cellB);
        
        if (!isNaN(numA) && !isNaN(numB)) {
          return ascending ? numA - numB : numB - numA;
        }
        
        // Fall back to string comparison
        return ascending 
          ? cellA.localeCompare(cellB)
          : cellB.localeCompare(cellA);
      });

      // Re-append sorted rows
      rows.forEach(row => tableElement.querySelector('tbody').appendChild(row));
    },

    filterRows: function(tableElement, predicate) {
      if (!tableElement) return;
      const rows = tableElement.querySelectorAll('tbody tr');
      rows.forEach((row, index) => {
        row.style.display = predicate(row, index) ? '' : 'none';
      });
    },

    searchRows: function(tableElement, searchTerm, columnIndex = 0) {
      const term = searchTerm.toLowerCase();
      this.filterRows(tableElement, (row) => {
        const cell = row.cells[columnIndex];
        return cell && cell.textContent.toLowerCase().includes(term);
      });
    }
  };

  window.Table = Table;

  // ======================================================================
  // UTILITY FUNCTIONS
  // ======================================================================

  window.Utils = {
    // Debounce function calls
    debounce: function(func, delay) {
      let timeoutId;
      return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func(...args), delay);
      };
    },

    // Throttle function calls
    throttle: function(func, limit) {
      let inThrottle;
      return function(...args) {
        if (!inThrottle) {
          func(...args);
          inThrottle = true;
          setTimeout(() => inThrottle = false, limit);
        }
      };
    },

    // Copy to clipboard
    copyToClipboard: function(text) {
      navigator.clipboard.writeText(text).then(() => {
        Toast.success('Copied to clipboard');
      }).catch(() => {
        Toast.error('Failed to copy');
      });
    },

    // Format currency
    formatCurrency: function(amount, currency = 'USD') {
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: currency
      }).format(amount);
    },

    // Format date
    formatDate: function(date, format = 'short') {
      const options = {
        short: { month: 'short', day: 'numeric', year: 'numeric' },
        long: { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' },
        time: { hour: '2-digit', minute: '2-digit' }
      };
      return new Date(date).toLocaleDateString('en-US', options[format] || options.short);
    }
  };

  // ======================================================================
  // PAGE INITIALIZATION
  // ======================================================================

  document.addEventListener('DOMContentLoaded', function() {
    // Auto-dismiss toasts
    const toasts = document.querySelectorAll('.toast');
    toasts.forEach(toast => {
      setTimeout(() => {
        toast.style.animation = 'fadeOut var(--transition-base)';
        setTimeout(() => toast.remove(), 250);
      }, 5000);
    });

    // Sidebar active state
    const navItems = document.querySelectorAll('.sidebar-nav-item');
    const currentPath = window.location.pathname;
    navItems.forEach(item => {
      const href = item.getAttribute('href');
      if (currentPath.includes(href.split('/').pop())) {
        item.classList.add('active');
      }
    });

    // Sidebar toggle (mobile)
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.getElementById('sidebar');
    if (sidebarToggle && sidebar) {
      sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('active');
      });

      // Close sidebar when clicking nav item
      document.querySelectorAll('.sidebar-nav-item').forEach(item => {
        item.addEventListener('click', () => {
          sidebar.classList.remove('active');
        });
      });
    }
  });

})();
