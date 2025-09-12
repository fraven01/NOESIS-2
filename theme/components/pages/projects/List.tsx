/**
 * Projects List page combining toolbar, table and pagination.
 */
import * as React from 'react';
import { DataToolbar } from '../../data/DataToolbar';
import { DataTable, Column, Row } from '../../data/DataTable';
import { DataPagination } from '../../data/DataPagination';

export interface Project {
  id: string;
  name: string;
  owner: string;
}

export interface ListProps {
  projects: Project[];
  loading?: boolean;
  error?: string;
}

export function List({ projects, loading = false, error }: ListProps) {
  const [search, setSearch] = React.useState('');
  const [selected, setSelected] = React.useState<string[]>([]);
  const [page, setPage] = React.useState(1);
  const pageSize = 10;

  const filtered = projects.filter((p) =>
    p.name.toLowerCase().includes(search.toLowerCase())
  );

  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize));
  const paged = filtered.slice((page - 1) * pageSize, page * pageSize);

  const columns: Column[] = [
    { key: 'name', label: 'Name' },
    { key: 'owner', label: 'Owner' },
  ];

  const rows: Row[] = paged.map((p) => ({
    id: p.id,
    name: p.name,
    owner: p.owner,
  }));

  if (loading) {
    return <p className="p-4">Loading…</p>;
  }
  if (error) {
    return <p className="p-4 text-danger">{error}</p>;
  }
  if (projects.length === 0) {
    return <p className="p-4">No projects found.</p>;
  }

  return (
    <div className="max-w-full">
      <DataToolbar
        onSearch={setSearch}
        hasSelection={selected.length > 0}
      />
      <DataTable
        columns={columns}
        rows={rows}
        onSelectionChange={setSelected}
      />
      <DataPagination
        page={page}
        pageCount={pageCount}
        onPageChange={setPage}
      />
    </div>
  );
}
