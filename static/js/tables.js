/**
 * Data Table Controller
 * Handles sorting, filtering, searching, and pagination for data tables
 */

(function() {
  'use strict';

  class DataTable {
    constructor(tableSelector, options = {}) {
      this.table = document.querySelector(tableSelector);
      if (!this.table) return;

      this.options = {
        sortable: true,
        searchable: true,
        filterable: true,
        perPage: 10,
        ...options
      };

      this.currentPage = 1;
      this.sortColumn = null;
      this.sortAscending = true;
      this.searchTerm = '';
      this.filters = {};
      this.allRows = [];

      this.init();
    }

    init() {
      this.allRows = Array.from(this.table.querySelectorAll('tbody tr'));
      this.bindSortHeaders();
      this.bindSearch();
      this.bindFilters();
    }

    bindSortHeaders() {
      if (!this.options.sortable) return;

      const headers = this.table.querySelectorAll('thead th');
      headers.forEach((header, index) => {
        header.style.cursor = 'pointer';
        header.addEventListener('click', () => this.sort(index));
        header.title = 'Click to sort';
      });
    }

    bindSearch() {
      if (!this.options.searchable) return;

      const searchInput = document.querySelector('[data-table-search="' + this.table.id + '"]');
      if (!searchInput) return;

      searchInput.addEventListener('input', (e) => {
        this.search(e.target.value);
      });
    }

    bindFilters() {
      if (!this.options.filterable) return;

      const filters = document.querySelectorAll('[data-table-filter="' + this.table.id + '"]');
      filters.forEach(filter => {
        filter.addEventListener('change', (e) => {
          const column = filter.getAttribute('data-column');
          const value = e.target.value;
          
          if (value) {
            this.filters[column] = value;
          } else {
            delete this.filters[column];
          }
          
          this.applyFilters();
        });
      });
    }

    sort(columnIndex) {
      if (this.sortColumn === columnIndex) {
        this.sortAscending = !this.sortAscending;
      } else {
        this.sortColumn = columnIndex;
        this.sortAscending = true;
      }

      this.displayRows();
      this.updateSortIndicators();
    }

    search(term) {
      this.searchTerm = term.toLowerCase();
      this.applyFilters();
    }

    applyFilters() {
      let visibleRows = this.allRows.filter(row => {
        // Apply search
        if (this.searchTerm) {
          const rowText = row.textContent.toLowerCase();
          if (!rowText.includes(this.searchTerm)) return false;
        }

        // Apply column filters
        for (let column in this.filters) {
          const cellIndex = parseInt(column);
          const filterValue = this.filters[column];
          const cellText = row.cells[cellIndex]?.textContent.toLowerCase() || '';
          
          if (!cellText.includes(filterValue.toLowerCase())) {
            return false;
          }
        }

        return true;
      });

      // Sort
      if (this.sortColumn !== null) {
        visibleRows.sort((a, b) => {
          const cellA = a.cells[this.sortColumn].textContent.trim();
          const cellB = b.cells[this.sortColumn].textContent.trim();

          const numA = parseFloat(cellA);
          const numB = parseFloat(cellB);

          let comparison = 0;
          if (!isNaN(numA) && !isNaN(numB)) {
            comparison = numA - numB;
          } else {
            comparison = cellA.localeCompare(cellB);
          }

          return this.sortAscending ? comparison : -comparison;
        });
      }

      // Update display
      this.tbody = this.table.querySelector('tbody');
      this.tbody.innerHTML = '';
      visibleRows.forEach(row => {
        this.tbody.appendChild(row.cloneNode(true));
      });

      // Re-bind event listeners on cloned rows
      this.bindRowActions();
    }

    displayRows() {
      this.applyFilters();
    }

    updateSortIndicators() {
      const headers = this.table.querySelectorAll('thead th');
      headers.forEach((header, index) => {
        if (index === this.sortColumn) {
          header.setAttribute('data-sort', this.sortAscending ? 'asc' : 'desc');
          header.style.opacity = '1';
        } else {
          header.removeAttribute('data-sort');
          header.style.opacity = '0.6';
        }
      });
    }

    bindRowActions() {
      // Bind edit, delete, etc. buttons on each row
      const editButtons = this.table.querySelectorAll('[data-action="edit"]');
      const deleteButtons = this.table.querySelectorAll('[data-action="delete"]');

      editButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.preventDefault();
          const rowData = this.getRowData(btn.closest('tr'));
          this.onEdit?.(rowData);
        });
      });

      deleteButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.preventDefault();
          if (confirm('Are you sure?')) {
            const rowData = this.getRowData(btn.closest('tr'));
            this.onDelete?.(rowData);
          }
        });
      });
    }

    getRowData(rowElement) {
      const cells = rowElement.querySelectorAll('td');
      const data = {};
      cells.forEach((cell, index) => {
        data[`col_${index}`] = cell.textContent.trim();
      });
      return data;
    }

    reset() {
      this.sortColumn = null;
      this.sortAscending = true;
      this.searchTerm = '';
      this.filters = {};
      this.displayRows();
    }
  }

  window.DataTable = DataTable;

  // ======================================================================
  // TABLE BUILDER HELPER
  // ======================================================================

  window.createTable = function(data, columns, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const table = document.createElement('table');
    table.className = 'data-table';

    // Header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    columns.forEach(col => {
      const th = document.createElement('th');
      th.textContent = col.label;
      th.dataset.field = col.field;
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Body
    const tbody = document.createElement('tbody');
    data.forEach(row => {
      const tr = document.createElement('tr');
      columns.forEach(col => {
        const td = document.createElement('td');
        const value = row[col.field];
        
        if (col.render) {
          td.innerHTML = col.render(value, row);
        } else {
          td.textContent = value;
        }
        
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    container.innerHTML = '';
    container.appendChild(table);

    return new DataTable(table);
  };

  // ======================================================================
  // AUTO-INITIALIZE DATA TABLES
  // ======================================================================

  document.addEventListener('DOMContentLoaded', function() {
    const tables = document.querySelectorAll('[data-table]');
    tables.forEach(table => {
      const tableId = table.id || table.getAttribute('data-table');
      window[tableId] = new DataTable('#' + tableId, {
        sortable: true,
        searchable: true,
        filterable: true
      });
    });
  });

})();
