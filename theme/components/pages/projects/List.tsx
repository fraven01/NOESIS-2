import * as React from "react";
import { DataToolbar } from "../../data/DataToolbar";
import { DataTable, Row } from "../../data/DataTable";
import { DataPagination } from "../../data/DataPagination";
import { EmptyState } from "../../ui/EmptyState";
import { ErrorState } from "../../ui/ErrorState";
import { Skeleton } from "../../ui/Skeleton";

/**
 * Generic projects list template combining toolbar, table and pagination.
 * @example
 * <List projects={[{id:'1', name:'Alpha'}]} />
 */
export interface ListProps {
  projects: Row[];
  status?: "idle" | "loading" | "error";
}

export const List: React.FC<ListProps> = ({ projects, status = "idle" }) => {
  const [search, setSearch] = React.useState("");
  const [selected, setSelected] = React.useState<string[]>([]);
  const [page, setPage] = React.useState(1);

  const filtered = React.useMemo(
    () => projects.filter((p) => p.name.toLowerCase().includes(search.toLowerCase())),
    [projects, search]
  );
  const pageSize = 5;
  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const pageItems = filtered.slice((page - 1) * pageSize, page * pageSize);

  React.useEffect(() => {
    setPage(1);
  }, [search]);

  return (
    <div className="flex flex-col gap-4">
      <DataToolbar
        search={search}
        onSearch={setSearch}
        selectedCount={selected.length}
        onDeleteSelected={() => setSelected([])}
      />
      {status === "loading" && <Skeleton className="h-24" />}
      {status === "error" && (
        <ErrorState title="Error" text="Something went wrong" />
      )}
      {status === "idle" && pageItems.length === 0 && (
        <EmptyState title="No projects" text="Create a project to get started" />
      )}
      {status === "idle" && pageItems.length > 0 && (
        <DataTable rows={pageItems} selected={selected} onSelectedChange={setSelected} />
      )}
      <DataPagination current={page} total={totalPages} onChange={setPage} />
    </div>
  );
};
