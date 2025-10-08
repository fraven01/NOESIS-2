import * as React from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { DataTable, Row } from "./DataTable";

const sampleRows: Row[] = [
  { id: "1", name: "Alpha" },
  { id: "2", name: "Bravo" },
  { id: "3", name: "Charlie" },
];

const meta: Meta<typeof DataTable> = {
  component: DataTable,
  args: {
    rows: sampleRows,
  },
};

export default meta;
type Story = StoryObj<typeof DataTable>;

export const Default: Story = {
  render: (args) => {
    const [selected, setSelected] = React.useState<string[]>([]);

    return (
      <DataTable
        {...args}
        selected={selected}
        onSelectedChange={(ids) => {
          setSelected(ids);
          args.onSelectedChange?.(ids);
        }}
      />
    );
  },
};
