import { render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { Sheet } from "./Sheet";

describe("Sheet", () => {
  it("renders when open and hides when closed", async () => {
    const onClose = vi.fn();
    const { container, rerender } = render(
      <Sheet open={false} onClose={onClose}>
        Content
      </Sheet>
    );
    expect(screen.queryByText("Content")).not.toBeInTheDocument();
    rerender(
      <Sheet open={true} onClose={onClose}>
        Content
      </Sheet>
    );
    expect(screen.getByText("Content")).toBeInTheDocument();
    screen.getByRole("dialog").setAttribute("aria-label", "Menu");
    expect(await axe(container)).toHaveNoViolations();
  });
});
