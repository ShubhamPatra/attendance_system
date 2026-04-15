/**
 * Component Behaviors & Animations
 * Reusable component logic, live updates, and animations
 */

(function() {
  'use strict';

  // ======================================================================
  // LIVE DETECTED STUDENTS LIST
  // ======================================================================

  class LiveDetectionList {
    constructor(containerId, options = {}) {
      this.container = document.getElementById(containerId);
      if (!this.container) return;

      this.options = {
        maxItems: 5,
        animationDuration: 300,
        autoRefresh: true,
        refreshInterval: 3000,
        ...options
      };

      this.items = [];
      this.init();
    }

    init() {
      this.container.innerHTML = '<div class="live-list"></div>';
      if (this.options.autoRefresh) {
        this.startAutoRefresh();
      }
    }

    startAutoRefresh() {
      this.refreshInterval = setInterval(() => {
        this.update();
      }, this.options.refreshInterval);
    }

    stopAutoRefresh() {
      clearInterval(this.refreshInterval);
    }

    addItem(data) {
      const item = {
        id: Math.random().toString(36).substr(2, 9),
        timestamp: new Date(),
        ...data
      };

      this.items.unshift(item);
      if (this.items.length > this.options.maxItems) {
        this.items.pop();
      }

      this.render();
      return item;
    }

    removeItem(id) {
      this.items = this.items.filter(item => item.id !== id);
      this.render();
    }

    update() {
      // Simulate real-time detection data
      const students = [
        { name: 'Elena Rodriguez', regNo: '#REG-2023-00142', status: 'Verified', time: 'Now' },
        { name: 'Marcus Chen', regNo: '#REG-2023-00155', status: 'Verifying', time: '5s ago' },
        { name: 'Amara Okafor', regNo: '#REG-2023-00211', status: 'Confirmed', time: '12s ago' }
      ];

      const random = students[Math.floor(Math.random() * students.length)];
      this.addItem(random);
    }

    render() {
      const list = this.container.querySelector('.live-list');
      if (!list) return;

      list.innerHTML = this.items.map(item => `
        <div class="live-item" style="animation: slideUp var(--transition-base);">
          <div class="live-item-avatar">
            <span class="material-symbols-outlined">person</span>
          </div>
          <div class="live-item-content">
            <div class="live-item-name">${item.name}</div>
            <div class="live-item-id">${item.regNo}</div>
          </div>
          <div class="live-item-status">
            <span class="badge badge-${item.status === 'Verified' ? 'success' : item.status === 'Verifying' ? 'warning' : 'secondary'}">
              ${item.status}
            </span>
          </div>
          <div class="live-item-time">${item.time}</div>
        </div>
      `).join('');
    }
  }

  window.LiveDetectionList = LiveDetectionList;

  // ======================================================================
  // PERCENTAGE CIRCLE ANIMATION
  // ======================================================================

  class PercentageCircle {
    constructor(containerId, percentage = 0, options = {}) {
      this.container = document.getElementById(containerId);
      if (!this.container) return;

      this.percentage = percentage;
      this.options = {
        size: 120,
        strokeWidth: 8,
        duration: 1500,
        color: 'var(--secondary)',
        backgroundColor: 'var(--surface-container-high)',
        ...options
      };

      this.render();
      this.animate();
    }

    render() {
      const size = this.options.size;
      const radius = (size - this.options.strokeWidth) / 2;
      const circumference = 2 * Math.PI * radius;

      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('width', size);
      svg.setAttribute('height', size);
      svg.setAttribute('viewBox', `0 0 ${size} ${size}`);

      // Background circle
      const bgCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      bgCircle.setAttribute('cx', size / 2);
      bgCircle.setAttribute('cy', size / 2);
      bgCircle.setAttribute('r', radius);
      bgCircle.setAttribute('fill', 'none');
      bgCircle.setAttribute('stroke', this.options.backgroundColor);
      bgCircle.setAttribute('stroke-width', this.options.strokeWidth);
      svg.appendChild(bgCircle);

      // Progress circle
      const progressCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      progressCircle.setAttribute('cx', size / 2);
      progressCircle.setAttribute('cy', size / 2);
      progressCircle.setAttribute('r', radius);
      progressCircle.setAttribute('fill', 'none');
      progressCircle.setAttribute('stroke', this.options.color);
      progressCircle.setAttribute('stroke-width', this.options.strokeWidth);
      progressCircle.setAttribute('stroke-dasharray', circumference);
      progressCircle.setAttribute('stroke-dashoffset', circumference);
      progressCircle.setAttribute('stroke-linecap', 'round');
      progressCircle.setAttribute('style', 'transform: rotate(-90deg); transform-origin: 50% 50%');
      svg.appendChild(progressCircle);

      this.container.innerHTML = '';
      this.container.appendChild(svg);
      this.progressCircle = progressCircle;
      this.circumference = circumference;
    }

    animate() {
      if (!this.progressCircle) return;

      const offset = this.circumference - (this.percentage / 100) * this.circumference;
      
      this.progressCircle.style.transition = `stroke-dashoffset ${this.options.duration}ms ease-out`;
      this.progressCircle.setAttribute('stroke-dashoffset', offset);
    }

    setPercentage(percentage) {
      this.percentage = Math.max(0, Math.min(100, percentage));
      this.animate();
    }
  }

  window.PercentageCircle = PercentageCircle;

  // ======================================================================
  // STAT CARD COUNTER
  // ======================================================================

  class StatCounter {
    constructor(elementSelector, endValue, options = {}) {
      this.element = document.querySelector(elementSelector);
      if (!this.element) return;

      this.startValue = 0;
      this.endValue = endValue;
      this.currentValue = this.startValue;
      this.options = {
        duration: 2000,
        easing: 'easeOutExpo',
        ...options
      };

      this.animate();
    }

    easeOutExpo(t) {
      return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
    }

    animate() {
      const startTime = Date.now();
      const duration = this.options.duration;

      const update = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easeProgress = this.easeOutExpo(progress);

        this.currentValue = Math.floor(
          this.startValue + (this.endValue - this.startValue) * easeProgress
        );

        this.element.textContent = this.currentValue.toLocaleString();

        if (progress < 1) {
          requestAnimationFrame(update);
        }
      };

      requestAnimationFrame(update);
    }
  }

  window.StatCounter = StatCounter;

  // ======================================================================
  // STATUS PULSE ANIMATION
  // ======================================================================

  class StatusIndicator {
    constructor(elementSelector, status = 'active', options = {}) {
      this.element = document.querySelector(elementSelector);
      if (!this.element) return;

      this.status = status;
      this.options = {
        colors: {
          active: 'var(--secondary)',
          inactive: 'var(--outline-variant)',
          processing: 'var(--warning)',
          error: 'var(--error)'
        },
        ...options
      };

      this.render();
    }

    render() {
      const color = this.options.colors[this.status] || this.options.colors.inactive;
      const shouldPulse = this.status === 'active' || this.status === 'processing' || this.status === 'error';

      this.element.className = `status-dot ${this.status}`;
      this.element.style.backgroundColor = color;
      
      if (shouldPulse) {
        this.element.style.animation = 'pulse 2s infinite';
      }
    }

    setStatus(status) {
      this.status = status;
      this.render();
    }
  }

  window.StatusIndicator = StatusIndicator;

  // ======================================================================
  // COLLAPSIBLE PANEL
  // ======================================================================

  class CollapsiblePanel {
    constructor(panelSelector, options = {}) {
      this.panel = document.querySelector(panelSelector);
      if (!this.panel) return;

      this.options = {
        startOpen: false,
        animationDuration: 300,
        ...options
      };

      this.isOpen = this.options.startOpen;
      this.init();
    }

    init() {
      const toggle = this.panel.querySelector('[data-toggle]');
      if (toggle) {
        toggle.addEventListener('click', () => this.toggle());
      }
    }

    toggle() {
      this.isOpen ? this.close() : this.open();
    }

    open() {
      const content = this.panel.querySelector('[data-content]');
      if (!content) return;

      this.isOpen = true;
      content.style.maxHeight = content.scrollHeight + 'px';
      content.style.opacity = '1';
      this.panel.classList.add('open');
    }

    close() {
      const content = this.panel.querySelector('[data-content]');
      if (!content) return;

      this.isOpen = false;
      content.style.maxHeight = '0';
      content.style.opacity = '0';
      this.panel.classList.remove('open');
    }
  }

  window.CollapsiblePanel = CollapsiblePanel;

  // ======================================================================
  // IMAGE LAZY LOADING
  // ======================================================================

  class LazyImage {
    constructor(selector = 'img[data-src]') {
      this.images = document.querySelectorAll(selector);
      this.init();
    }

    init() {
      if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
          entries.forEach(entry => {
            if (entry.isIntersecting) {
              this.loadImage(entry.target);
              observer.unobserve(entry.target);
            }
          });
        });

        this.images.forEach(img => observer.observe(img));
      } else {
        // Fallback for older browsers
        this.images.forEach(img => this.loadImage(img));
      }
    }

    loadImage(img) {
      const src = img.getAttribute('data-src');
      if (src) {
        img.src = src;
        img.removeAttribute('data-src');
      }
    }
  }

  window.LazyImage = LazyImage;

  // ======================================================================
  // FADE IN OBSERVER
  // ======================================================================

  class FadeInOnScroll {
    constructor(selector = '[data-fade-in]') {
      this.elements = document.querySelectorAll(selector);
      this.init();
    }

    init() {
      if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
          entries.forEach(entry => {
            if (entry.isIntersecting) {
              entry.target.style.animation = 'fadeIn var(--transition-base)';
              observer.unobserve(entry.target);
            }
          });
        });

        this.elements.forEach(el => observer.observe(el));
      }
    }
  }

  window.FadeInOnScroll = FadeInOnScroll;

  // ======================================================================
  // PAGE INITIALIZATION
  // ======================================================================

  document.addEventListener('DOMContentLoaded', function() {
    // Initialize lazy images
    new LazyImage();

    // Initialize fade-in animations
    new FadeInOnScroll();

    // Add CSS for collapsible panels
    const style = document.createElement('style');
    style.textContent = `
      [data-content] {
        max-height: 0;
        overflow: hidden;
        opacity: 0;
        transition: max-height var(--transition-base), opacity var(--transition-base);
      }

      .collapsible.open [data-content] {
        opacity: 1;
      }
    `;
    document.head.appendChild(style);
  });

})();
