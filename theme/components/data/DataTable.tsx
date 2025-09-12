/**
 * DataTable renders data in a table with sortable columns and selectable rows.
 */
import * as React from 'react';

export interface Column {
  key: string;
  label: string;
}

export interface Row {
  id: string;
  [key: string]: React.ReactNode;
}

export interface DataTableProps {
  columns: Column[];
  rows: Row[];
  onSelectionChange?: (ids: string[]) => void;
}

export function DataTable({ columns, rows, onSelectionChange }: DataTableProps) {
  const [selected, setSelected] = React.useState<string[]>([]);
  const [sort, setSort] = React.useState<{ key: string; dir: 'asc' | 'desc' } | null>(null);

  const toggleAll = () => {
    if (selected.length === rows.length) {
      setSelected([]);
    } else {
      setSelected(rows.map((r) => r.id));
    }
  };

  const toggleRow = (id: string) => {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  React.useEffect(() => {
    onSelectionChange?.(selected);
  }, [selected, onSelectionChange]);

  const sortedRows = React.useMemo(() => {
    if (!sort) return rows;
    return [...rows].sort((a, b) => {
      const av = String(a[sort.key] ?? '');
      const bv = String(b[sort.key] ?? '');
      return sort.dir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
    });
  }, [rows, sort]);

  const selectAllChecked = selected.length === rows.length && rows.length > 0;
  const selectAllMixed = selected.length > 0 && selected.length < rows.length;

  const handleSort = (key: string) => {
    setSort((prev) =>
      prev && prev.key === key
        ? { key, dir: prev.dir === 'asc' ? 'desc' : 'asc' }
        : { key, dir: 'asc' }
    );
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left" role="grid">
        <thead className="border-b">
          <tr>
            <th className="p-2">
              <input
                type="checkbox"
                aria-label="Select all"
                checked={selectAllChecked}
                aria-checked={selectAllMixed ? 'mixed' : selectAllChecked ? 'true' : 'false'}
                onChange={toggleAll}
              />
            </th>
            {columns.map((col) => (
              <th
                key={col.key}
                className="p-2"
                aria-sort={
                  sort?.key === col.key
                    ? sort.dir === 'asc'
                      ? 'ascending'
                      : 'descending'
                    : 'none'
                }
              >
                <button
                  type="button"
                  onClick={() => handleSort(col.key)}
                  className="flex items-center gap-1"
                >
                  {col.label}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedRows.map((row) => (
            <tr key={row.id} className="border-b">
              <td className="p-2">
                <input
                  type="checkbox"
                  aria-label={`Select ${row.id}`}
                  checked={selected.includes(row.id)}
                  onChange={() => toggleRow(row.id)}
                />
              </td>
              {columns.map((col) => (
                <td key={col.key} className="p-2">
                  {row[col.key] as React.ReactNode}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
