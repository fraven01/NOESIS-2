import * as React from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { Table, TableHead, TableRow, SortableHeader, TableBody, TableCell } from "./Table";

const meta: Meta<typeof Table> = { component: Table };
export default meta;
type Story = StoryObj<typeof Table>;

const ExampleTable: React.FC = () => (
  <Table caption="Users">
    <TableHead>
      <TableRow>
        <SortableHeader id="name">Name</SortableHeader>
        <SortableHeader id="age">Age</SortableHeader>
      </TableRow>
    </TableHead>
    <TableBody>
      <TableRow>
        <TableCell>Alice</TableCell>
        <TableCell>30</TableCell>
      </TableRow>
      <TableRow>
        <TableCell>Bob</TableCell>
        <TableCell>25</TableCell>
      </TableRow>
    </TableBody>
  </Table>
);

export const Default: Story = { render: () => <ExampleTable /> };
export const States: Story = { render: () => <ExampleTable /> };
export const Dark: Story = {
  render: () => (
    <div className="dark p-4 bg-bg text-fg">
      <ExampleTable />
    </div>
  ),
};
export const RTL: Story = {
  render: () => (
    <div dir="rtl">
      <ExampleTable />
    </div>
  ),
};
