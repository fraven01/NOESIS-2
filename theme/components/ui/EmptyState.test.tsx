import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { EmptyState } from "./EmptyState";

describe("EmptyState", () => {
  it("renders content and handles CTA click", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    const { container } = render(
      <EmptyState title="No items" text="Add one" cta={{ label: "Add", onClick }} />
    );
    expect(
      screen.getByRole("heading", { name: "No items" })
    ).toBeInTheDocument();
    expect(screen.getByText("Add one")).toBeInTheDocument();
    const button = screen.getByRole("button", { name: "Add" });
    await user.click(button);
    expect(onClick).toHaveBeenCalledTimes(1);
    expect(await axe(container)).toHaveNoViolations();
  });
});
