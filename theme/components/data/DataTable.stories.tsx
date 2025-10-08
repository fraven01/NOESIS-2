import { useState, type ComponentProps, type FC } from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { DataTable } from "./DataTable";

const meta: Meta<typeof DataTable> = {
  component: DataTable,
};

export default meta;

type Story = StoryObj<typeof DataTable>;

const rows = [
  { id: "1", name: "Documents" },
  { id: "2", name: "Policies" },
  { id: "3", name: "Training" },
];

type StatefulProps = Omit<ComponentProps<typeof DataTable>, "rows" | "selected" | "onSelectedChange"> & {
  initialSelected?: string[];
};

const StatefulTable: FC<StatefulProps> = ({ initialSelected = [], ...props }) => {
  const [selected, setSelected] = useState<string[]>(initialSelected);
  return (
    <DataTable
      {...props}
      rows={rows}
      selected={selected}
      onSelectedChange={setSelected}
    />
  );
};

export const Default: Story = {
  render: (args) => <StatefulTable {...args} />,
};

export const PreselectedRow: Story = {
  render: (args) => <StatefulTable {...args} initialSelected={["2"]} />,
};
