import { useState, type FC, type PropsWithChildren } from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { DataToolbar, type DataToolbarProps } from "./DataToolbar";
import { Button } from "../ui/Button";

const meta: Meta<typeof DataToolbar> = {
  component: DataToolbar,
};

export default meta;

type Story = StoryObj<typeof DataToolbar>;

type StatefulProps = Omit<DataToolbarProps, "onSearch" | "onDeleteSelected"> & {
  initialSelected?: number;
} & PropsWithChildren;

const StatefulToolbar: FC<StatefulProps> = ({ initialSelected = 0, children, ...props }) => {
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState(initialSelected);

  return (
    <div className="flex flex-col gap-4">
      <DataToolbar
        {...props}
        search={search}
        onSearch={setSearch}
        selectedCount={selected}
        onDeleteSelected={() => setSelected(0)}
      >
        {children ?? (
          <div className="flex flex-col gap-2 text-sm">
            <span className="font-medium">Filters</span>
            <label className="flex items-center gap-2">
              <input type="checkbox" /> Active
            </label>
            <label className="flex items-center gap-2">
              <input type="checkbox" /> Archived
            </label>
          </div>
        )}
      </DataToolbar>
      <div className="flex items-center gap-2 text-sm text-muted">
        <span>Selected rows: {selected}</span>
        <Button size="sm" variant="secondary" onClick={() => setSelected((count) => count + 1)}>
          Add selection
        </Button>
        <Button size="sm" variant="ghost" onClick={() => setSelected((count) => Math.max(0, count - 1))}>
          Remove selection
        </Button>
      </div>
    </div>
  );
};

export const Default: Story = {
  render: (args) => <StatefulToolbar {...args} />,
};

export const WithSelection: Story = {
  render: (args) => <StatefulToolbar {...args} initialSelected={3} />,
};
