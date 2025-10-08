import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { useState, type FC } from "react";
import { DataToolbar, type DataToolbarProps } from "./DataToolbar";

const StatefulToolbar: FC<DataToolbarProps> = (props) => {
  const [search, setSearch] = useState(props.search);
  return (
    <DataToolbar
      {...props}
      search={search}
      onSearch={(value) => {
        props.onSearch(value);
        setSearch(value);
      }}
    />
  );
};

describe("DataToolbar", () => {
  it("handles search, filter and bulk actions", async () => {
    const user = userEvent.setup();
    const onSearch = vi.fn();
    const onDeleteSelected = vi.fn();
    const { container } = render(
      <StatefulToolbar
        search=""
        onSearch={onSearch}
        selectedCount={2}
        onDeleteSelected={onDeleteSelected}
      />
    );

    const searchBox = screen.getByRole("textbox", { name: /search/i });
    await user.type(searchBox, "abc");
    expect(onSearch).toHaveBeenLastCalledWith("abc");

    const filterButton = screen.getByRole("button", { name: /filter/i });
    await user.click(filterButton);
    expect(await screen.findByText(/no filters/i)).toBeInTheDocument();

    const deleteButton = screen.getByRole("button", { name: /delete selected/i });
    await user.click(deleteButton);
    expect(onDeleteSelected).toHaveBeenCalledTimes(1);

    expect(await axe(container)).toHaveNoViolations();
  });

  it("hides bulk actions when nothing is selected", () => {
    render(
      <StatefulToolbar
        search="query"
        onSearch={() => {}}
        selectedCount={0}
        onDeleteSelected={() => {}}
      />
    );

    expect(
      screen.queryByRole("button", { name: /delete selected/i })
    ).not.toBeInTheDocument();
  });
});
