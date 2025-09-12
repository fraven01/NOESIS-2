import type { Meta, StoryObj } from "@storybook/react";
import { Dashboard } from "./Dashboard";
import { Card } from "../ui/Card";
import { Skeleton } from "../ui/Skeleton";
import { EmptyState } from "../ui/EmptyState";
import { ErrorState } from "../ui/ErrorState";

const meta: Meta<typeof Dashboard> = {
  title: "Pages/Dashboard",
  component: Dashboard,
};
export default meta;
type Story = StoryObj<typeof Dashboard>;

const SkeletonSlot = () => (
  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
    {Array.from({ length: 4 }).map((_, index) => (
      <Card key={index}>
        <Skeleton className="h-24" />
      </Card>
    ))}
  </div>
);

const EmptySlot = () => (
  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
    <Card>
      <EmptyState title="Empty" text="Nothing here" />
    </Card>
  </div>
);

const ErrorSlot = () => (
  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
    <Card>
      <ErrorState title="Error" text="Something went wrong" />
    </Card>
  </div>
);

export const Default: Story = {
  args: {
    KpiSlot: SkeletonSlot,
    ActivitySlot: EmptySlot,
    QuickLinksSlot: EmptySlot,
  },
};

export const Loading: Story = {
  args: {
    KpiSlot: SkeletonSlot,
    ActivitySlot: SkeletonSlot,
    QuickLinksSlot: SkeletonSlot,
  },
};

export const Empty: Story = {
  args: {
    KpiSlot: EmptySlot,
    ActivitySlot: EmptySlot,
    QuickLinksSlot: EmptySlot,
  },
};

export const Error: Story = {
  args: {
    KpiSlot: ErrorSlot,
    ActivitySlot: ErrorSlot,
    QuickLinksSlot: ErrorSlot,
  },
};

export const Dark: Story = {
  render: (args) => (
    <div className="dark bg-bg p-4 text-fg">
      <Dashboard {...args} />
    </div>
  ),
  args: {
    KpiSlot: SkeletonSlot,
    ActivitySlot: EmptySlot,
    QuickLinksSlot: EmptySlot,
  },
};

export const RTL: Story = {
  render: (args) => (
    <div dir="rtl">
      <Dashboard {...args} />
    </div>
  ),
  args: {
    KpiSlot: SkeletonSlot,
    ActivitySlot: EmptySlot,
    QuickLinksSlot: EmptySlot,
  },
};
