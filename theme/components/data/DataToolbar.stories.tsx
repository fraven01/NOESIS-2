import * as React from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { DataToolbar } from "./DataToolbar";

const meta: Meta<typeof DataToolbar> = {
  component: DataToolbar,
  args: {
    search: "",
    selectedCount: 0,
  },
};

export default meta;
type Story = StoryObj<typeof DataToolbar>;

export const Default: Story = {
  render: (args) => {
    const [search, setSearch] = React.useState(args.search ?? "");
    const [selected, setSelected] = React.useState(args.selectedCount ?? 0);

    return (
      <div className="flex flex-col gap-4">
        <DataToolbar
          {...args}
          search={search}
          selectedCount={selected}
          onSearch={(value) => {
            setSearch(value);
            args.onSearch?.(value);
          }}
          onDeleteSelected={() => {
            setSelected(0);
            args.onDeleteSelected?.();
          }}
        >
          <div className="text-sm">Example filter content</div>
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
  },
};
