import type { Meta, StoryObj } from "@storybook/react";
import { Detail } from "./Detail";
import { Card } from "../../ui/Card";
import { EmptyState } from "../../ui/EmptyState";
import { ErrorState } from "../../ui/ErrorState";
import { Skeleton } from "../../ui/Skeleton";

const meta: Meta<typeof Detail> = {
  title: "Pages/Projects/Detail",
  component: Detail,
};
export default meta;
type Story = StoryObj<typeof Detail>;

const SkeletonSlot = () => (
  <Card>
    <Skeleton className="h-24" />
  </Card>
);

const EmptySlot = () => (
  <Card>
    <EmptyState title="Empty" text="Nothing here" />
  </Card>
);

const ErrorSlot = () => (
  <Card>
    <ErrorState title="Error" text="Something went wrong" />
  </Card>
);

export const Default: Story = {
  args: {
    title: "Project Alpha",
    status: "Active",
  },
};

export const Loading: Story = {
  args: {
    title: "Project Alpha",
    status: "Active",
    OverviewSlot: SkeletonSlot,
    HistorySlot: SkeletonSlot,
    AttachmentsSlot: SkeletonSlot,
    SecondarySlot: SkeletonSlot,
  },
};

export const Empty: Story = {
  args: {
    title: "Project Alpha",
    status: "Active",
    OverviewSlot: EmptySlot,
    HistorySlot: EmptySlot,
    AttachmentsSlot: EmptySlot,
    SecondarySlot: EmptySlot,
  },
};

export const Error: Story = {
  args: {
    title: "Project Alpha",
    status: "Active",
    OverviewSlot: ErrorSlot,
    HistorySlot: ErrorSlot,
    AttachmentsSlot: ErrorSlot,
    SecondarySlot: ErrorSlot,
  },
};

export const Dark: Story = {
  render: (args) => (
    <div className="dark bg-bg p-4 text-fg">
      <Detail {...args} />
    </div>
  ),
  args: {
    title: "Project Alpha",
    status: "Active",
  },
};

export const RTL: Story = {
  render: (args) => (
    <div dir="rtl">
      <Detail {...args} />
    </div>
  ),
  args: {
    title: "Project Alpha",
    status: "Active",
  },
};
