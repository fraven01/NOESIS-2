import { useState, type ComponentProps, type FC } from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { DataPagination } from "./DataPagination";

const meta: Meta<typeof DataPagination> = {
  component: DataPagination,
  args: {
    total: 5,
  },
};

export default meta;

type Story = StoryObj<typeof DataPagination>;

type StatefulProps = Omit<ComponentProps<typeof DataPagination>, "current" | "onChange"> & {
  initialPage?: number;
};

const StatefulPagination: FC<StatefulProps> = ({ initialPage = 1, ...props }) => {
  const [page, setPage] = useState(initialPage);
  return (
    <DataPagination
      {...props}
      current={page}
      onChange={(next) => {
        setPage(next);
      }}
    />
  );
};

export const Default: Story = {
  render: (args) => <StatefulPagination {...args} />,
};

export const ManyPages: Story = {
  args: {
    total: 10,
  },
  render: (args) => <StatefulPagination {...args} />,
};
