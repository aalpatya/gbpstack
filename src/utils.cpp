/**************************************************************************************/
// Copyright (c) 2024 Aalok Patwardhan (a.patwardhan21@imperial.ac.uk)
// This code is licensed (see LICENSE for details)
/**************************************************************************************/
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <gbp/utils.h>
#include <gbp/globals.h>
extern Globals globals;

Eigen::MatrixXd conv2d(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel )
{
    int n = input.rows();           //input size
    int k = kernel.rows();          // kernel size
    int p = (int) (k-1)/2;          // padding either side
    int padded_sz = n + 2*p;
    Eigen::MatrixXd input_pad = Eigen::MatrixXd::Zero(padded_sz, padded_sz);
    input_pad.block(p, p, n, n) = input;
    Eigen::MatrixXd output_mat = Eigen::MatrixXd::Zero(n, n);

    double normalization = kernel.sum();
    if ( normalization < 1E-6 ) normalization=1;
#pragma omp parallel for
    for (int row = 0; row < n; row++ ){
      for (int col = 0; col < n; col++ ){
        output_mat.coeffRef(row,col) = ( input_pad.block(row, col, k, k).cwiseProduct(kernel) ).sum();
      }
    }
   
    return output_mat/normalization;
}
// Draws a texture in 3D space with pro parameters...
void DrawTexturePro3D(Texture2D texture, Rectangle sourceRec, Rectangle destRec, Vector3 origin, float rotation, float posZ, Color tint)
{
    // Check if texture is valid
    if (texture.id > 0)
    {
        float width = (float)texture.width;
        float height = (float)texture.height;
        
        bool flipX = false;

        if (sourceRec.width < 0) { flipX = true; sourceRec.width *= -1; }
        if (sourceRec.height < 0) sourceRec.y -= sourceRec.height;

        rlEnableTexture(texture.id);
        rlPushMatrix();
            rlTranslatef(destRec.x, destRec.y, 0.0f);
            rlRotatef(rotation, 0.0f, 0.0f, 1.0f);
            rlTranslatef(-origin.x, -origin.y, -origin.z);

            rlBegin(RL_QUADS);
                rlColor4ub(tint.r, tint.g, tint.b, tint.a);
                rlNormal3f(0.0f, 0.0f, 1.0f);                          // Normal vector pointing towards viewer

                // Bottom-left corner for texture and quad
                if (flipX) rlTexCoord2f((sourceRec.x + sourceRec.width)/width, sourceRec.y/height);
                else rlTexCoord2f(sourceRec.x/width, sourceRec.y/height);
                rlVertex3f(0.0f, 0.0f, posZ);
                
                // Bottom-right corner for texture and quad
                if (flipX) rlTexCoord2f((sourceRec.x + sourceRec.width)/width, (sourceRec.y + sourceRec.height)/height);
                else rlTexCoord2f(sourceRec.x/width, (sourceRec.y + sourceRec.height)/height);
                rlVertex3f(0.0f, destRec.height, posZ);

                // Top-right corner for texture and quad
                if (flipX) rlTexCoord2f(sourceRec.x/width, (sourceRec.y + sourceRec.height)/height);
                else rlTexCoord2f((sourceRec.x + sourceRec.width)/width, (sourceRec.y + sourceRec.height)/height);
                rlVertex3f(destRec.width, destRec.height, posZ);

                // Top-left corner for texture and quad
                if (flipX) rlTexCoord2f(sourceRec.x/width, sourceRec.y/height);
                else rlTexCoord2f((sourceRec.x + sourceRec.width)/width, sourceRec.y/height);
                rlVertex3f(destRec.width, 0.0f, posZ);
            rlEnd();
        rlPopMatrix();
        rlDisableTexture();
    }
}

void draw_info(uint32_t iter_cnt, uint32_t time_cnt, MODES_LIST mode, MOUSE_MODES_LIST mouse_mode, bool HELP){
    static std::map<MODES_LIST, const char*> MODES_MAP = {
        {SimNone,""},
        {Timestep,"Timestep"},
        {OneTimestep,"One Timestep"},
        {Iterate,"Synchronous Iteration"},
        {Help, "Help"},
        {Junction, "Junction"},
    };

    static std::map<MOUSE_MODES_LIST, const char*> MOUSE_MODES_MAP = {
        {MouseNone,""},
        {Obstacle,"Add Obstacle"},
        {Attractor,"Add Attractor"},
        {Swarm,"Swarm"},
        {Draw,"Draw shape"},
        {AddRobotStart, "Add Robot Start"},
        {AddRobotGoal, "Add Robot Goal"},
        {Eraser, "Eraser"},
        {RobotStartStop, "Start/Stop Robot"}
    };    

    int info_box_h = 100, info_box_w = 180;;
    int info_box_x = 10, info_box_y = globals.SCREEN_SZ-40-info_box_h;
    // DrawRectangle( info_box_x, info_box_y, info_box_w, info_box_h, Fade(SKYBLUE, 0.5f));
    // DrawRectangleLines( info_box_x, info_box_y, info_box_w, info_box_h, BLUE);

    // DrawText(TextFormat("Goal horizon: %f seconds", globals.T_HORIZON), info_box_x + 10, info_box_y + 10, 10, BLACK);
    // DrawText(TextFormat("Iterations per timestep: %i", N_ITERS_PER_TIMESTEP), info_box_x + 10, info_box_y + 50, 10, BLACK);
    // DrawText("HELP: PRESS H", info_box_x + 10, info_box_y + 70, 20, BLACK);

    DrawRectangle(0, globals.SCREEN_SZ-30, globals.SCREEN_SZ, 30, BLACK);
    DrawText(TextFormat("Time: %.1f s", time_cnt*globals.TIMESTEP), 5, globals.SCREEN_SZ-20, 20, DARKGREEN);
    // DrawText(TextFormat("%s", MODES_MAP[mode]), globals.SCREEN_SZ/4., globals.SCREEN_SZ-20, 20, DARKGREEN);
    // DrawText(TextFormat("%s", MOUSE_MODES_MAP[mouse_mode]), 2*globals.SCREEN_SZ/3., globals.SCREEN_SZ-20, 20, GREEN);
    DrawFPS(20,10);

    if (mode==Help){
        int info_box_h = 500, info_box_w = 500;
        int info_box_x = globals.SCREEN_SZ/2 - info_box_w/2, info_box_y = globals.SCREEN_SZ/2 - info_box_h/2;
        DrawRectangle( info_box_x, info_box_y, info_box_w, info_box_h, Fade(SKYBLUE, 0.5f));
        DrawRectangleLines( info_box_x, info_box_y, info_box_w, info_box_h, BLACK);

        int offset = 10;
        DrawText("R - Reset", info_box_x + 10, info_box_y + offset, 20, BLACK);
        DrawText("I - Iterate GBP", info_box_x + 10, info_box_y + (offset+=35), 20, BLACK);
        DrawText("T - Run simulation", info_box_x + 10, info_box_y + (offset+=35), 20, BLACK);
        DrawText("H - Toggle help", info_box_x + 10, info_box_y + (offset+=35), 20, BLACK);
        DrawText("O - Add obstacle", info_box_x + 10, info_box_y + (offset+=35), 20, BLACK);
        DrawText("G - Show goal and futures", info_box_x + 10, info_box_y + (offset+=35), 20, BLACK);
        DrawText("F - Show obstacle factors", info_box_x + 10, info_box_y + (offset+=35), 20, BLACK);
        DrawText("A - Add robot", info_box_x + 10, info_box_y + (offset+=35), 20, BLACK);
        DrawText("Q - Quit", info_box_x + 10, info_box_y + (offset+=35), 20, BLACK);

    }


}

std::vector<Eigen::VectorXd> linspace(Eigen::VectorXd low, Eigen::VectorXd high, int num_elems, bool upper_inclusive){
    // If upper_inclusive is true, returns a vector of at least size 2 ([low, high])
    // otherwise returns vector of at least size 1 ([low])
    std::vector<Eigen::VectorXd> return_vector{};
    int num_dims = low.size();
    num_elems = (upper_inclusive==true) ? max(num_elems, 2) : max(num_elems, 1);
    float num_steps = (upper_inclusive==true) ? num_elems-1 : num_elems;
    for (int i = 0; i < num_elems; i++){
        Eigen::VectorXd newvec = low + i*(high-low)/num_steps;
        return_vector.push_back(newvec);
    }    

    return return_vector;
}

float mahalanobis_distSquared(const Eigen::VectorXd& z, const Eigen::VectorXd& mu, const Eigen::MatrixXd& Lam){
    float d2 = (z-mu).transpose() * Lam * (z-mu);
    return d2;
}

std::vector<int> get_variable_list(int lookahead_horizon, int lookahead_multiple){
    std::vector<int> var_list{};
    int N = 1 + int(0.5*(-1 + sqrt(1 + 8*(float)lookahead_horizon/(float)lookahead_multiple)));

#if 0
    for (int i=0; i<lookahead_horizon; i++){
        var_list.push_back(i);
    }
#else    
    for (int i=0; i<lookahead_multiple*(N+1); i++){
        int section = int(i/lookahead_multiple);
        int f = (i - section*lookahead_multiple + lookahead_multiple/2.*section)*(section+1);
        if (f>=lookahead_horizon){
            var_list.push_back(lookahead_horizon);
            break;
        }
        var_list.push_back(f);
    }
#endif

    return var_list;
};

// Draw circle within an image
// Copied and edited from raylib.h
void ImageDrawCircleFilled(Image *dst, int centerX, int centerY, int radius, Color color)
{
    int x = 0, yy = radius;
    int decesionParameter = 3 - 2*radius;

    while (yy >= x)
    {
        for (int y=0; y<yy; y++){
        ImageDrawPixel(dst, centerX + x, centerY + y, color);
        ImageDrawPixel(dst, centerX - x, centerY + y, color);
        ImageDrawPixel(dst, centerX + x, centerY - y, color);
        ImageDrawPixel(dst, centerX - x, centerY - y, color);
        ImageDrawPixel(dst, centerX + y, centerY + x, color);
        ImageDrawPixel(dst, centerX - y, centerY + x, color);
        ImageDrawPixel(dst, centerX + y, centerY - x, color);
        ImageDrawPixel(dst, centerX - y, centerY - x, color);
        }
        x++;

        if (decesionParameter > 0)
        {
            yy--;
            decesionParameter = decesionParameter + 4*(x - yy) + 10;
        }
        else decesionParameter = decesionParameter + 4*x + 6;
    }
}

Color color_grad(Color start, Color end, float amount){
    auto [r1,g1,b1,a1] = start;
    auto [r2,g2,b2,a2] = end;
    amount = min(1.f, amount);
    unsigned char r = (int) ((float) r1 + (float) (r2 - r1) * amount);
    unsigned char g = (int) ((float) g1 + (float) (g2 - g1) * amount);
    unsigned char b = (int) ((float) b1 + (float) (b2 - b1) * amount);
    unsigned char a = (int) ((float) a1 + (float) (a2 - a1) * amount);

    return (Color){r,g,b,a};
}

void write_csv(std::string filename, std::vector<std::pair<std::string, std::vector<int>>> dataset){
    // Make a CSV file with one or more columns of integer values
    // Each column of data is represented by the pair <column name, column data>
    //   as std::pair<std::string, std::vector<int>>
    // The dataset is represented as a vector of these columns
    // Note that all columns should be the same size
    
    // Create an output filestream object
    std::ofstream myFile(filename);
    
    // Send column names to the stream
    for(int j = 0; j < dataset.size(); ++j)
    {
        myFile << dataset.at(j).first;
        if(j != dataset.size() - 1) myFile << ","; // No comma at end of line
    }
    myFile << "\n";
    
    // Send data to the stream
    for(int i = 0; i < dataset.at(0).second.size(); ++i)
    {
        for(int j = 0; j < dataset.size(); ++j)
        {
            myFile << dataset.at(j).second.at(i);
            if(j != dataset.size() - 1) myFile << ","; // No comma at end of line
        }
        myFile << "\n";
    }
    
    // Close the file
    myFile.close();    
}

std::map<int, std::map<int, std::vector<float>>> read_csv(std::string filename){
   // File pointer
    std::ifstream fin(filename);
    std::cout << "Reading " << filename << "\n";
    if (fin.fail()){
        std::cout << "FAIL\n";
    } else { std::cout << "GOOD\n";}

    // Read the Data from the file
    // as String Vector
    // timestamps = [t0, t1, ...] = [[r0, r1,...], ...] = [[[param0, param1, ...], ...], ...]
    std::map<int, std::map<int, std::vector<float>>> timestamps;
    std::string line, elem_str, temp;
    bool skip = true;

    if (skip) std::getline(fin, line);

    while (std::getline(fin, line)) {
        std::vector<std::vector<float>> robots{};
        std::vector<std::string> robot_data_str{};

        // read an entire row and
        // store it in a string variable 'line'

        // used for breaking words
        std::stringstream s(line);

        // split line into words
        while (std::getline(s, elem_str, ',')) {
            // add all the column data
            // of a row to a vector
            robot_data_str.push_back(elem_str);
        }
        int ts = std::stoi(robot_data_str[0]);  //timestep
        int rid = std::stoi(robot_data_str[1]);  //timestep
        for (int i=2; i<robot_data_str.size(); i++){
            timestamps[ts][rid].push_back(
                std::stof(robot_data_str[i])
            );
        }
    }
    fin.close();

    return timestamps;    
};

std::map<int, std::vector<std::tuple<int, float, float, float, float>>> read_record(std::string filename)
{

    // File pointer
    std::ifstream fin(filename);
    std::cout << "Reading " << filename << "\n";
    if (fin.fail()){
        std::cout << "FAIL\n";
    } else { std::cout << "GOOD\n";}

    // // Read the Data from the file
    // // as String Vector
    std::map<int, std::vector<std::tuple<int, float, float, float, float>>> history;
    std::string line, elem_str, temp;
    bool skip = true;

    if (skip) std::getline(fin, line);

    while (std::getline(fin, line)) {
        std::vector<std::tuple<int, float, float, float, float>> robot_data{};
        std::vector<std::string> robot_data_str{};

        // read an entire row and
        // store it in a string variable 'line'

        // used for breaking words
        std::stringstream s(line);

        // split line into words
        while (std::getline(s, elem_str, ',')) {
            // add all the column data
            // of a row to a vector
            robot_data_str.push_back(elem_str);
        }
        int ts = std::stoi(robot_data_str[0]);  //timestep
        history[ts].push_back(std::make_tuple(std::stoi(robot_data_str[1]), // rid
                                            std::stof(robot_data_str[2]),   // x0
                                            std::stof(robot_data_str[3]), // y0
                                            std::stof(robot_data_str[4]), // y1
                                            std::stof(robot_data_str[5]))); // y1

    }
    fin.close();

    return history;
}


////
// Draw a 2D text in 3D space
void DrawText3D(Font font, const char *text, Vector3 position, float fontSize, float fontSpacing, float lineSpacing, bool backface, Color tint)
{
    int length = TextLength(text);          // Total length in bytes of the text, scanned by codepoints in loop

    float textOffsetY = 0.0f;               // Offset between lines (on line break '\n')
    float textOffsetX = 0.0f;               // Offset X to next character to draw

    float scale = fontSize/(float)font.baseSize;

    for (int i = 0; i < length;)
    {
        // Get next codepoint from byte string and glyph index in font
        int codepointByteCount = 0;
        int codepoint = GetCodepoint(&text[i], &codepointByteCount);
        int index = GetGlyphIndex(font, codepoint);

        // NOTE: Normally we exit the decoding sequence as soon as a bad byte is found (and return 0x3f)
        // but we need to draw all of the bad bytes using the '?' symbol moving one byte
        if (codepoint == 0x3f) codepointByteCount = 1;

        if (codepoint == '\n')
        {
            // NOTE: Fixed line spacing of 1.5 line-height
            // TODO: Support custom line spacing defined by user
            textOffsetY += scale + lineSpacing/(float)font.baseSize*scale;
            textOffsetX = 0.0f;
        }
        else
        {
            if ((codepoint != ' ') && (codepoint != '\t'))
            {
                DrawTextCodepoint3D(font, codepoint, (Vector3){ position.x + textOffsetX, position.y, position.z + textOffsetY }, fontSize, backface, tint);
            }

            if (font.glyphs[index].advanceX == 0) textOffsetX += (float)(font.recs[index].width + fontSpacing)/(float)font.baseSize*scale;
            else textOffsetX += (float)(font.glyphs[index].advanceX + fontSpacing)/(float)font.baseSize*scale;
        }

        i += codepointByteCount;   // Move text bytes counter to next codepoint
    }
}

// Draw codepoint at specified position in 3D space
void DrawTextCodepoint3D(Font font, int codepoint, Vector3 position, float fontSize, bool backface, Color tint)
{
    // Character index position in sprite font
    // NOTE: In case a codepoint is not available in the font, index returned points to '?'
    int index = GetGlyphIndex(font, codepoint);
    float scale = fontSize/(float)font.baseSize;

    // Character destination rectangle on screen
    // NOTE: We consider charsPadding on drawing
    position.x += (float)(font.glyphs[index].offsetX - font.glyphPadding)/(float)font.baseSize*scale;
    position.z += (float)(font.glyphs[index].offsetY - font.glyphPadding)/(float)font.baseSize*scale;

    // Character source rectangle from font texture atlas
    // NOTE: We consider chars padding when drawing, it could be required for outline/glow shader effects
    Rectangle srcRec = { font.recs[index].x - (float)font.glyphPadding, font.recs[index].y - (float)font.glyphPadding,
                         font.recs[index].width + 2.0f*font.glyphPadding, font.recs[index].height + 2.0f*font.glyphPadding };

    float width = (float)(font.recs[index].width + 2.0f*font.glyphPadding)/(float)font.baseSize*scale;
    float height = (float)(font.recs[index].height + 2.0f*font.glyphPadding)/(float)font.baseSize*scale;

    if (font.texture.id > 0)
    {
        const float x = 0.0f;
        const float y = 0.0f;
        const float z = 0.0f;

        // normalized texture coordinates of the glyph inside the font texture (0.0f -> 1.0f)
        const float tx = srcRec.x/font.texture.width;
        const float ty = srcRec.y/font.texture.height;
        const float tw = (srcRec.x+srcRec.width)/font.texture.width;
        const float th = (srcRec.y+srcRec.height)/font.texture.height;

        // if (true) DrawCubeWiresV((Vector3){ position.x + width/2, position.y, position.z + height/2}, (Vector3){ width, LETTER_BOUNDRY_SIZE, height }, LETTER_BOUNDRY_COLOR);

        rlCheckRenderBatchLimit(4 + 4*backface);
        rlSetTexture(font.texture.id);

        rlPushMatrix();
            rlTranslatef(position.x, position.y, position.z);

            rlBegin(RL_QUADS);
                rlColor4ub(tint.r, tint.g, tint.b, tint.a);

                // Front Face
                rlNormal3f(0.0f, 1.0f, 0.0f);                                   // Normal Pointing Up
                rlTexCoord2f(tx, ty); rlVertex3f(x,         y, z);              // Top Left Of The Texture and Quad
                rlTexCoord2f(tx, th); rlVertex3f(x,         y, z + height);     // Bottom Left Of The Texture and Quad
                rlTexCoord2f(tw, th); rlVertex3f(x + width, y, z + height);     // Bottom Right Of The Texture and Quad
                rlTexCoord2f(tw, ty); rlVertex3f(x + width, y, z);              // Top Right Of The Texture and Quad

                if (backface)
                {
                    // Back Face
                    rlNormal3f(0.0f, -1.0f, 0.0f);                              // Normal Pointing Down
                    rlTexCoord2f(tx, ty); rlVertex3f(x,         y, z);          // Top Right Of The Texture and Quad
                    rlTexCoord2f(tw, ty); rlVertex3f(x + width, y, z);          // Top Left Of The Texture and Quad
                    rlTexCoord2f(tw, th); rlVertex3f(x + width, y, z + height); // Bottom Left Of The Texture and Quad
                    rlTexCoord2f(tx, th); rlVertex3f(x,         y, z + height); // Bottom Right Of The Texture and Quad
                }
            rlEnd();
        rlPopMatrix();

        rlSetTexture(0);
    }
}
void DrawEllipse3D(Vector3 centerPos, Vector3 radius, int rings, int slices, Color color)
{
    rlPushMatrix();
        // NOTE: Transformation is applied in inverse order (scale -> translate)
        rlTranslatef(centerPos.x, centerPos.y, centerPos.z);
        rlScalef(radius.x, radius.y, radius.z);

        rlBegin(RL_LINES);
            rlColor4ub(color.r, color.g, color.b, color.a);

            for (int i = 0; i < (rings + 2); i++)
            {
                for (int j = 0; j < slices; j++)
                {
                    rlVertex3f(cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*i))*sinf(DEG2RAD*(360.0f*j/slices)),
                               sinf(DEG2RAD*(270 + (180.0f/(rings + 1))*i)),
                               cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*i))*cosf(DEG2RAD*(360.0f*j/slices)));
                    rlVertex3f(cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1)))*sinf(DEG2RAD*(360.0f*(j + 1)/slices)),
                               sinf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1))),
                               cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1)))*cosf(DEG2RAD*(360.0f*(j + 1)/slices)));

                    rlVertex3f(cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1)))*sinf(DEG2RAD*(360.0f*(j + 1)/slices)),
                               sinf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1))),
                               cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1)))*cosf(DEG2RAD*(360.0f*(j + 1)/slices)));
                    rlVertex3f(cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1)))*sinf(DEG2RAD*(360.0f*j/slices)),
                               sinf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1))),
                               cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1)))*cosf(DEG2RAD*(360.0f*j/slices)));

                    rlVertex3f(cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1)))*sinf(DEG2RAD*(360.0f*j/slices)),
                               sinf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1))),
                               cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1)))*cosf(DEG2RAD*(360.0f*j/slices)));
                    rlVertex3f(cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*i))*sinf(DEG2RAD*(360.0f*j/slices)),
                               sinf(DEG2RAD*(270 + (180.0f/(rings + 1))*i)),
                               cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*i))*cosf(DEG2RAD*(360.0f*j/slices)));
                }
            }
        rlEnd();
    rlPopMatrix();
}
void DrawEllipse(Vector3 center, Vector2 radius, Vector3 rotationAxis, float rotationAngle, Color color)
{
    rlPushMatrix();
        rlTranslatef(center.x, center.y, center.z);
        rlRotatef(rotationAngle, rotationAxis.x, rotationAxis.y, rotationAxis.z);

        rlBegin(RL_LINES);
            for (int i = 0; i < 360; i += 10)
            {
                rlColor4ub(color.r, color.g, color.b, color.a);

                rlVertex3f(sinf(DEG2RAD*i)*radius.x, 0.0f, cosf(DEG2RAD*i)*radius.y);
                rlVertex3f(sinf(DEG2RAD*(i + 10))*radius.x, 0.0f, cosf(DEG2RAD*(i + 10))*radius.y);
            }
        rlEnd();
    rlPopMatrix();
}