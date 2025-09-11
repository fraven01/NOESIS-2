import type { Meta, StoryObj } from "@storybook/react";
import { Dashboard, DashboardProps } from "./Dashboard";

const meta: Meta<typeof Dashboard> = {
  title: "Pages/Dashboard",
  component: Dashboard,
};
export default meta;
type Story = StoryObj<typeof Dashboard>;

const kpis: DashboardProps["kpis"] = [
  { label: "Users", value: "1.2k" },
  { label: "Sessions", value: "3.4k" },
  { label: "Errors", value: "23" },
  { label: "Uptime", value: "99.9%" },
];

const activities: DashboardProps["activities"] = [
  { id: "1", text: "User signed up", status: "success" },
  { id: "2", text: "Server restarted", status: "warning" },
  { id: "3", text: "Error reported", status: "error" },
];

const quickLinks: DashboardProps["quickLinks"] = [
  { label: "Projects", href: "#" },
  { label: "Documents", href: "#" },
  { label: "Settings", href: "#" },
];

export const Default: Story = {
  args: { kpis, activities, quickLinks },
};

export const Loading: Story = {
  args: { kpis: [], activities: [], quickLinks: [], loading: true },
};

export const Empty: Story = {
  args: { kpis: kpis.slice(0, 3), activities: [], quickLinks: [] },
};

export const Error: Story = {
  args: {
    kpis: [],
    activities: [],
    quickLinks: [],
    error: "Failed to load data",
  },
};

export const Dark: Story = {
  render: (args) => (
    <div className="dark bg-bg p-4 text-fg">
      <Dashboard {...args} />
    </div>
  ),
  args: { kpis, activities, quickLinks },
};

export const RTL: Story = {
  render: (args) => (
    <div dir="rtl">
      <Dashboard {...args} />
    </div>
  ),
  args: { kpis, activities, quickLinks },
};
