import * as React from "react";
import type { FC, PropsWithChildren } from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { DataToolbar, type DataToolbarProps } from "./DataToolbar";

type StatefulProps = PropsWithChildren<{
  search?: DataToolbarProps["search"];
  selectedCount?: DataToolbarProps["selectedCount"];
  initialSearch?: string;
  initialSelected?: number;
  onSearch?: DataToolbarProps["onSearch"];
  onDeleteSelected?: DataToolbarProps["onDeleteSelected"];
}>;

const StatefulToolbar: FC<StatefulProps> = ({
  search: controlledSearch,
  selectedCount: controlledSelected,
  initialSearch,
  initialSelected,
  onSearch,
  onDeleteSelected,
  children,
}) => {
  const [search, setSearch] = React.useState(controlledSearch ?? initialSearch ?? "");
  const [selected, setSelected] = React.useState(controlledSelected ?? initialSelected ?? 0);

  React.useEffect(() => {
    if (controlledSearch !== undefined) {
      setSearch(controlledSearch);
    }
  }, [controlledSearch]);

  React.useEffect(() => {
    if (controlledSelected !== undefined) {
      setSelected(controlledSelected);
    }
  }, [controlledSelected]);

  return (
    <div className="flex flex-col gap-4">
      <DataToolbar
        search={search}
        selectedCount={selected}
        onSearch={(value) => {
          setSearch(value);
          onSearch?.(value);
        }}
        onDeleteSelected={() => {
          setSelected(0);
          onDeleteSelected?.();
        }}
      >
        {children ?? <div className="text-sm">Example filter content</div>}
      </DataToolbar>
      <button
        type="button"
        className="self-start rounded-lg border border-muted px-3 py-1 text-sm"
        onClick={() => setSelected((count) => count + 1)}
      >
        Simulate selection
      </button>
    </div>
  );
};

const meta = {
  component: DataToolbar,
  args: {
    search: "",
    selectedCount: 0,
  },
} satisfies Meta<typeof DataToolbar>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: (args) => <StatefulToolbar {...args} />,
};
