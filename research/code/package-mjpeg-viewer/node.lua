gl.setup(NATIVE_WIDTH, NATIVE_HEIGHT)

util.no_globals()

local layout = "auto"

local font = resource.load_font "OpenSans-Bold.ttf"
local black = resource.create_colored_texture(0, 0, 0, 1)

local next_image_load = sys.now()
local image_per_second = 2

local function Cam(config)
    local img = resource.create_colored_texture(0, 0, 0, 0)
    local next_img
    local last_update = sys.now()

    local function update(file)
        if sys.now() < next_image_load then
            print("discarding due to image rate")
        elseif not next_img then
            print 'loading next frame for cam'
            next_img = resource.load_image(file)
            if image_per_second == "unlimited" then
                next_image_load = sys.now()
            else
                next_image_load = sys.now() + 1 / image_per_second
            end
        else
            print 'discarding frame'
        end
    end

    local function get()
        if next_img then
            local state = next_img:state()
            if state == 'loaded' then
                local old_img
                old_img, img, next_img = img, next_img, nil
                old_img:dispose()
                last_update = sys.now()
            elseif state == 'error' then
                next_img:dispose()
                next_img = nil
            end
        end

        return {
            age = sys.now() - last_update,
            name = config.name,
            img = img,
        }
    end

    return {
        update = update;
        get = get;
    }
end

local cams = {}

node.event("content_update", function(filename, file)
    if filename:sub(1, 13) == 'zz-cam-frame-' then
        local cam_num = tonumber(filename:sub(14,16))
        cams[cam_num].update(file)

    end
end)

util.json_watch("config.json", function(config)
    cams = {}

    for idx, cam in ipairs(config.cams) do
        cams[idx] = Cam(cam)
    end
    print(string.format("updated config. %d cams", #cams))

    layout = config.layout
    image_per_second = config.image_per_second

    node.gc()
end)

function node.render()
    gl.clear(0, 0, 0, 1)
    local split_x, split_y
    if layout == "auto" then
        if #cams > 9 then
            split_x = 4
            split_y = 4
        elseif #cams > 4 then
            split_x = 3
            split_y = 3
        elseif #cams > 1 then
            split_x = 2
            split_y = 2
        else
            split_x = 1
            split_y = 1
        end
    else
        split_x = layout[1]
        split_y = layout[2]
    end

    local w = math.floor(WIDTH / split_x)
    local h = math.floor(HEIGHT / split_y)
    for idx = 1, #cams do
        local x = (idx-1) % split_x
        local y = math.floor((idx-1) / split_x)

        local cam = cams[idx].get()

        local name = string.format("cam %d", idx)
        if #cam.name > 0 then
            name = cam.name
        end

        local status
        if cam.age > 3 then
            status = string.format("last updated %ds ago", cam.age)
        else
            status = "live"
        end

        local info = string.format("%s - %s", name, status)
        local x1, y1, x2, y2 = x * w, y * h, (x+1) * w, (y+1) * h

        util.draw_correct(cam.img, x1, y1, x2, y2)

        local size = 20
        local w = font:width(info, size)
        black:draw(x1, y2-size-6, x1+w+6, y2, 0.5)
        font:write(x1+3, y2-size-3, info, size, 1,1,1,1)
    end
end
