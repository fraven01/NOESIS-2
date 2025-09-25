import * as React from "react";
import { Badge } from "../../ui/Badge";
import { Button } from "../../ui/Button";
import { Card } from "../../ui/Card";
import { EmptyState } from "../../ui/EmptyState";
import { Dialog } from "../../ui/Dialog";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../../ui/Tabs";

/**
 * Project detail template with header, tabs and secondary column.
 * @example
 * <Detail title="Alpha" status="Active" />
 */
export interface DetailProps {
  title: string;
  status: string;
  primaryActionLabel?: string;
  OverviewSlot?: React.ComponentType;
  HistorySlot?: React.ComponentType;
  AttachmentsSlot?: React.ComponentType;
  SecondarySlot?: React.ComponentType;
}

const DefaultOverviewSlot: React.FC = () => (
  <Card>
    <EmptyState title="Overview" text="No details" />
  </Card>
);

const DefaultHistorySlot: React.FC = () => (
  <Card>
    <EmptyState title="History" text="No history" />
  </Card>
);

const DropzonePlaceholder: React.FC = () => (
  <div
    role="button"
    tabIndex={0}
    aria-label="Upload files"
    className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-muted p-8 text-center text-muted focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
  >
    <p>Drag and drop files here or click to upload</p>
  </div>
);

const DefaultSecondarySlot: React.FC = () => (
  <Card>
    <EmptyState title="Sidebar" text="Nothing here" />
  </Card>
);

export const Detail: React.FC<DetailProps> = ({
  title,
  status,
  primaryActionLabel = "Edit",
  OverviewSlot = DefaultOverviewSlot,
  HistorySlot = DefaultHistorySlot,
  AttachmentsSlot = DropzonePlaceholder,
  SecondarySlot = DefaultSecondarySlot,
}) => {
  const [open, setOpen] = React.useState(false);
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <h1 className="text-2xl font-bold">{title}</h1>
        <div className="flex items-center gap-2">
          <Badge>{status}</Badge>
          <Button onClick={() => setOpen(true)}>{primaryActionLabel}</Button>
        </div>
      </header>
      <div className="grid gap-8 lg:grid-cols-[2fr_1fr]">
        <div>
          <Tabs defaultValue="overview">
            <TabsList className="mb-4">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="history">History</TabsTrigger>
              <TabsTrigger value="attachments">Attachments</TabsTrigger>
            </TabsList>
            <TabsContent value="overview">
              <OverviewSlot />
            </TabsContent>
            <TabsContent value="history">
              <HistorySlot />
            </TabsContent>
            <TabsContent value="attachments">
              <AttachmentsSlot />
            </TabsContent>
          </Tabs>
        </div>
        <aside className="hidden lg:block">
          <SecondarySlot />
        </aside>
      </div>
      <Dialog open={open} onClose={() => setOpen(false)} ariaLabel="Primary Action Dialog">
        <p className="mb-4">Dialog content</p>
        <Button onClick={() => setOpen(false)}>Close</Button>
      </Dialog>
    </div>
  );
};
