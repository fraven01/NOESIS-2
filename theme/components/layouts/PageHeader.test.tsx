import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { PageHeader } from "./PageHeader";
import { Button } from "../ui/Button";

test("breadcrumb marks current page and opens mobile actions", async () => {
  const user = userEvent.setup();
  render(
    <PageHeader
      title="Titel"
      breadcrumbs={[
        { href: "/", label: "Home" },
        { href: "/projects", label: "Projekte" },
      ]}
      actions={<Button size="sm">Speichern</Button>}
    />
  );
  const current = screen.getByText("Projekte");
  expect(current).toHaveAttribute("aria-current", "page");
  const trigger = screen.getByLabelText(/aktionen/i);
  trigger.focus();
  await user.keyboard("{Enter}");
  const dialog = await screen.findByRole("dialog");
  expect(
    within(dialog).getByRole("button", { name: "Speichern" })
  ).toBeVisible();
  await user.keyboard("{Escape}");
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
});

test("has no accessibility violations", async () => {
  const { container } = render(
    <PageHeader
      title="Titel"
      breadcrumbs={[
        { href: "/", label: "Home" },
        { href: "/projects", label: "Projekte" },
      ]}
      actions={<Button size="sm">Speichern</Button>}
    />
  );
  expect(await axe(container)).toHaveNoViolations();
});
