import * as React from "react";
import * as Checkbox from "@radix-ui/react-checkbox";
import { CheckedState } from "@radix-ui/react-checkbox";
import { Check } from "lucide-react";

/**
 * Data table with selectable rows and sortable columns.
 * @example
 * <DataTable rows={rows} selected={selected} onSelectedChange={setSelected} />
 */
export interface Row {
  id: string;
  name: string;
}

export interface DataTableProps {
  rows: Row[];
  selected: string[];
  onSelectedChange: (ids: string[]) => void;
}

export const DataTable: React.FC<DataTableProps> = ({ rows, selected, onSelectedChange }) => {
  const [sortDir, setSortDir] = React.useState<"ascending" | "descending" | "none">("none");

  const sorted = React.useMemo(() => {
    const data = [...rows];
    if (sortDir === "ascending") {
      data.sort((a, b) => a.name.localeCompare(b.name));
    } else if (sortDir === "descending") {
      data.sort((a, b) => b.name.localeCompare(a.name));
    }
    return data;
  }, [rows, sortDir]);

  const allIds = rows.map((r) => r.id);
  const allSelected = selected.length === allIds.length && allIds.length > 0;
  const someSelected = selected.length > 0 && !allSelected;
  const headerState: CheckedState = allSelected ? true : someSelected ? "indeterminate" : false;

  const toggleAll = (state: CheckedState) => {
    if (state) {
      onSelectedChange(allIds);
    } else {
      onSelectedChange([]);
    }
  };

  const toggleOne = (id: string, state: CheckedState) => {
    if (state) {
      onSelectedChange([...selected, id]);
    } else {
      onSelectedChange(selected.filter((s) => s !== id));
    }
  };

  const nextSort = () => {
    setSortDir((d) => (d === "none" ? "ascending" : d === "ascending" ? "descending" : "none"));
  };

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-left text-sm">
        <thead>
          <tr>
            <th className="px-4 py-2">
              <Checkbox.Root
                aria-label="Select all"
                checked={headerState}
                onCheckedChange={toggleAll}
                className="flex h-4 w-4 items-center justify-center rounded border border-muted bg-bg data-[state=checked]:bg-accent data-[state=indeterminate]:bg-accent focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              >
                <Checkbox.Indicator>
                  <Check className="h-3 w-3 text-bg" />
                </Checkbox.Indicator>
              </Checkbox.Root>
            </th>
            <th
              scope="col"
              className="cursor-pointer select-none px-4 py-2"
              aria-sort={sortDir}
              onClick={nextSort}
            >
              Name
            </th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr key={row.id} className="border-t border-muted">
              <td className="px-4 py-2">
                <Checkbox.Root
                  aria-label={`Select ${row.name}`}
                  checked={selected.includes(row.id)}
                  onCheckedChange={(s) => toggleOne(row.id, s)}
                  className="flex h-4 w-4 items-center justify-center rounded border border-muted bg-bg data-[state=checked]:bg-accent focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                >
                  <Checkbox.Indicator>
                    <Check className="h-3 w-3 text-bg" />
                  </Checkbox.Indicator>
                </Checkbox.Root>
              </td>
              <td className="px-4 py-2">{row.name}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
