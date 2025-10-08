import * as React from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { DataPagination } from "./DataPagination";

const meta: Meta<typeof DataPagination> = {
  component: DataPagination,
  args: {
    current: 1,
    total: 5,
  },
};

export default meta;
type Story = StoryObj<typeof DataPagination>;

export const Default: Story = {
  render: (args) => {
    const [page, setPage] = React.useState(args.current ?? 1);

    return (
      <DataPagination
        {...args}
        current={page}
        onChange={(value) => {
          setPage(value);
          args.onChange?.(value);
        }}
      />
    );
  },
};
