import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { Tooltip } from "./Tooltip";
import { Button } from "./Button";

describe("Tooltip", () => {
  it("appears on hover and focus", async () => {
    const user = userEvent.setup();
    const { container } = render(
      <Tooltip content="Info">
        <Button>Trigger</Button>
      </Tooltip>
    );
    const trigger = screen.getByRole("button", { name: /trigger/i });
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
    await user.hover(trigger);
    expect(await screen.findByRole("tooltip")).toHaveTextContent("Info");
    await user.unhover(trigger);
    await waitFor(() =>
      expect(screen.queryByRole("tooltip")).not.toBeInTheDocument()
    );
    await user.tab();
    expect(await screen.findByRole("tooltip")).toHaveTextContent("Info");
    await user.tab();
    await waitFor(() =>
      expect(screen.queryByRole("tooltip")).not.toBeInTheDocument()
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});
