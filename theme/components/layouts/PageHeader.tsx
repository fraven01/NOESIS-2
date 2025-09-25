import * as React from "react";
import * as Popover from "@radix-ui/react-popover";
import { MoreHorizontal } from "lucide-react";
import { Button } from "../ui/Button";

/**
 * Page header with breadcrumbs, title, optional meta information and actions.
 * @example
 * <PageHeader title="Projects" breadcrumbs={[{href:'/', label:'Home'}, {href:'/projects', label:'Projects'}]} meta="5 items" actions={<Button size="sm">New</Button>} />
 */
export interface BreadcrumbItem {
  href: string;
  label: string;
}

export interface PageHeaderProps {
  title: string;
  breadcrumbs?: BreadcrumbItem[];
  meta?: React.ReactNode;
  actions?: React.ReactNode;
}

export const PageHeader: React.FC<PageHeaderProps> = ({
  title,
  breadcrumbs,
  meta,
  actions,
}) => (
  <header className="border-b bg-bg">
    {breadcrumbs && (
      <nav className="px-4 py-2" aria-label="Breadcrumb">
        <ol className="flex flex-wrap text-sm">
          {breadcrumbs.map((item, i) => {
            const isLast = i === breadcrumbs.length - 1;
            return (
              <li key={item.href} className="flex items-center gap-1">
                {i > 0 && <span className="text-muted">/</span>}
                {isLast ? (
                  <span aria-current="page" className="truncate">
                    {item.label}
                  </span>
                ) : (
                  <a href={item.href} className="text-accent hover:underline">
                    {item.label}
                  </a>
                )}
              </li>
            );
          })}
        </ol>
      </nav>
    )}
    <div className="flex flex-wrap items-center gap-2 px-4 py-2">
      <div className="min-w-0 flex-1">
        <h1 className="text-lg font-semibold leading-tight">
          {title}
        </h1>
        {meta && <div className="text-sm text-muted">{meta}</div>}
      </div>
      {actions && (
        <div className="flex items-center gap-2">
          <div className="hidden sm:flex items-center gap-2">{actions}</div>
          <div className="sm:hidden">
            <Popover.Root>
              <Popover.Trigger asChild>
                <Button aria-label="Aktionen" variant="ghost" size="sm">
                  <MoreHorizontal className="h-4 w-4" />
                </Button>
              </Popover.Trigger>
              <Popover.Content
                sideOffset={4}
                align="end"
                className="z-50 rounded-xl border border-muted bg-bg p-2 shadow-sm"
              >
                <div className="flex flex-col gap-2">{actions}</div>
              </Popover.Content>
            </Popover.Root>
          </div>
        </div>
      )}
    </div>
  </header>
);

export default PageHeader;

