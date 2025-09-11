import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { Tooltip } from "./Tooltip";
import { Button } from "./Button";

describe("Tooltip", () => {
  it("shows on hover and is accessible", async () => {
    const user = userEvent.setup();
    const { container } = render(
      <Tooltip content="Info">
        <Button>Trigger</Button>
      </Tooltip>
    );
    const trigger = screen.getByRole("button", { name: /trigger/i });
    await user.hover(trigger);
    expect(await screen.findByRole("tooltip")).toHaveTextContent("Info");
    expect(await axe(container)).toHaveNoViolations();
  });
});
