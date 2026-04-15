(() => {
  const dataScript = document.getElementById('dashboard-data');
  const initialData = dataScript ? JSON.parse(dataScript.textContent || '{}') : {};
  const root = document.documentElement;
  const body = document.body;
  const themeToggle = document.querySelector('[data-theme-toggle]');
  const sidebarToggle = document.querySelector('[data-sidebar-toggle]');
  const sidebarBackdrop = document.querySelector('[data-sidebar-backdrop]');
  const feed = document.getElementById('activityFeed');
  const liveState = document.querySelector('[data-live-state]');
  const chartCanvas = document.getElementById('attendanceTrendChart');

  const themeKey = 'autoattendance.theme';

  const getPreferredTheme = () => {
    const stored = window.localStorage.getItem(themeKey);
    if (stored === 'dark' || stored === 'light') return stored;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  };

  const applyTheme = (theme) => {
    root.setAttribute('data-theme', theme);
    window.localStorage.setItem(themeKey, theme);
  };

  applyTheme(getPreferredTheme());

  themeToggle?.addEventListener('click', () => {
    const nextTheme = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    applyTheme(nextTheme);
  });

  const setSidebarOpen = (open) => {
    body.dataset.sidebarOpen = open ? 'true' : 'false';
  };

  sidebarToggle?.addEventListener('click', () => {
    setSidebarOpen(body.dataset.sidebarOpen !== 'true');
  });

  sidebarBackdrop?.addEventListener('click', () => setSidebarOpen(false));

  const animateCounters = () => {
    document.querySelectorAll('[data-counter]').forEach((node) => {
      const target = Number(node.dataset.target || '0');
      const decimals = Number(node.dataset.decimals || '0');
      const duration = 800;
      const start = performance.now();

      const tick = (now) => {
        const progress = Math.min((now - start) / duration, 1);
        const value = target * (0.2 + 0.8 * progress);
        node.textContent = decimals > 0 ? value.toFixed(decimals) : Math.round(value).toString();
        if (progress < 1) {
          window.requestAnimationFrame(tick);
        } else {
          node.textContent = decimals > 0 ? target.toFixed(decimals) : target.toString();
        }
      };

      window.requestAnimationFrame(tick);
    });
  };

  const renderChart = () => {
    if (!chartCanvas || typeof Chart === 'undefined') return null;
    const labels = JSON.parse(chartCanvas.dataset.chartLabels || '[]');
    const values = JSON.parse(chartCanvas.dataset.chartValues || '[]');
    return new Chart(chartCanvas, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Attendance',
          data: values,
          borderColor: '#4f46e5',
          backgroundColor: 'rgba(79, 70, 229, 0.14)',
          fill: true,
          tension: 0.35,
          pointRadius: 4,
          pointHoverRadius: 6,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(15, 23, 42, 0.92)',
            padding: 12,
          },
        },
        scales: {
          x: { grid: { display: false } },
          y: { beginAtZero: true, ticks: { precision: 0 } },
        },
      },
    });
  };

  const statusLabel = (status) => {
    const value = String(status || 'unknown').replaceAll('_', ' ');
    return value.charAt(0).toUpperCase() + value.slice(1);
  };

  const statusClass = (status) => `status-${String(status || 'unknown').toLowerCase()}`;

  const prepActivityItem = (item) => {
    const row = document.createElement('div');
    row.className = 'activity-item';
    row.dataset.recordId = item.id || '';
    row.innerHTML = `
      <div class="activity-item__time">${item.time || '--:--'}</div>
      <div class="activity-item__body">
        <div class="activity-item__title">${item.student_name || 'Unknown student'}</div>
        <div class="activity-item__meta">${[item.course_code, item.course_name].filter(Boolean).join(' · ')}</div>
      </div>
      <div class="activity-item__status ${statusClass(item.status)}">${statusLabel(item.status)}</div>
    `;
    return row;
  };

  const prependActivity = (item) => {
    if (!feed) return;
    const existing = feed.querySelector(`[data-record-id="${CSS.escape(String(item.id || ''))}"]`);
    if (existing) existing.remove();
    feed.prepend(prepActivityItem(item));
    while (feed.children.length > 20) {
      feed.removeChild(feed.lastElementChild);
    }
  };

  animateCounters();
  const chart = renderChart();

  const socket = typeof io === 'function' ? io() : null;
  if (!socket) {
    liveState && (liveState.textContent = 'Offline');
    liveState && liveState.classList.remove('badge-success');
    liveState && liveState.classList.add('badge-warning');
    return;
  }

  socket.on('stats_refresh', (payload) => {
    window.__dashboardStats = payload;
    if (payload?.attendance_trend && chart) {
      chart.data.labels = payload.attendance_trend.labels || [];
      chart.data.datasets[0].data = payload.attendance_trend.values || [];
      chart.update();
    }
  });

  socket.on('attendance_update', (payload) => {
    if (payload?.recent_checkin) {
      prependActivity(payload.recent_checkin);
    }
  });

  socket.on('connect', () => {
    if (!liveState) return;
    liveState.textContent = 'Connected';
    liveState.classList.remove('badge-warning');
    liveState.classList.add('badge-success');
  });

  socket.on('disconnect', () => {
    if (!liveState) return;
    liveState.textContent = 'Offline';
    liveState.classList.remove('badge-success');
    liveState.classList.add('badge-warning');
  });

  setInterval(() => {
    socket.emit('stats_refresh');
  }, 30000);

  if (Object.keys(initialData || {}).length) {
    window.__dashboardStats = initialData;
  }
})();
