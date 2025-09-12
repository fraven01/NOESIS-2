import type { Meta, StoryObj } from "@storybook/react";
import { PageHeader } from "./PageHeader";
import { Button } from "../ui/Button";

const meta: Meta<typeof PageHeader> = {
  component: PageHeader,
  title: "Layouts/PageHeader",
};

export default meta;
type Story = StoryObj<typeof PageHeader>;

const crumbs = [
  { href: "/", label: "Home" },
  { href: "/projects", label: "Projekte" },
  { href: "#", label: "Aktuell" },
];

const actions = (
  <>
    <Button size="sm">Speichern</Button>
    <Button size="sm" variant="secondary">
      Abbrechen
    </Button>
  </>
);

export const Default: Story = {
  render: () => (
    <PageHeader
      title="Titel"
      breadcrumbs={crumbs}
      meta={<span>Meta</span>}
      actions={actions}
    />
  ),
};

export const Loading: Story = {
  render: () => (
    <PageHeader
      title="Lade"
      breadcrumbs={crumbs}
      meta={<span>Lädt…</span>}
      actions={
        <Button size="sm" loading>
          Speichern
        </Button>
      }
    />
  ),
};

export const Empty: Story = {
  render: () => <PageHeader title="Ohne Extras" />, 
};

export const Error: Story = {
  render: () => (
    <PageHeader
      title="Fehler"
      breadcrumbs={crumbs}
      meta={<span className="text-danger">Fehler beim Laden</span>}
      actions={actions}
    />
  ),
};

export const Dark: Story = {
  parameters: { backgrounds: { default: "dark" } },
  render: () => (
    <div className="dark">
      <PageHeader
        title="Titel"
        breadcrumbs={crumbs}
        meta={<span>Meta</span>}
        actions={actions}
      />
    </div>
  ),
};

export const RTL: Story = {
  parameters: { direction: "rtl" },
  render: () => (
    <div dir="rtl">
      <PageHeader
        title="عنوان"
        breadcrumbs={crumbs}
        meta={<span>بيانات</span>}
        actions={actions}
      />
    </div>
  ),
};

